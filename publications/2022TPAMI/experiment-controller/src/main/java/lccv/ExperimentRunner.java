package lccv;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimenterFrontend;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.processes.ProcessIDNotRetrievableException;
import ai.libs.jaicore.processes.ProcessUtil;

public class ExperimentRunner implements IExperimentSetEvaluator {

	private static final Logger logger = LoggerFactory.getLogger("experimenter");

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {
		try {

			/* get configuration */
			Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
			int openmlid = Integer.valueOf(keys.get("openmlid"));
			int seed = Integer.valueOf(keys.get("seed"));
			int numpipelines = Integer.valueOf(keys.get("numpipelines"));
			String probDP = keys.get("prob_dp");
			String probFP = keys.get("prob_fp");
			String train_size = keys.get("train_size");
			String algo = keys.get("algorithm");
			String timeout = keys.get("timeout");
			logger.info("\topenmlid: {}", openmlid);
			logger.info("\ttrain_size: {}", train_size);
			logger.info("\talgo: {}", algo);
			logger.info("\tseed: {}", seed);
			logger.info("\tnumpipelines: {}", numpipelines);
			logger.info("\ttimeout: {}", timeout);

			/* run python experiment */
			String options = "--dataset_id=" + openmlid + " --train_size=" + train_size + " --algorithm=" + algo + " --seed=" + seed + " --num_pipelines=" + numpipelines + " --timeout=" + timeout + " --prob_dp=" + probDP + " --prob_fp=" + probFP;
			JsonNode results = getPythonExperimentResults(options);
			logger.info("Obtained result json node: {}", results);

			/* write results */
			Map<String, Object> map = new HashMap<>();
			map.put("chosenmodel", results.get(0).asText());
			map.put("errorrate", results.get(1).asDouble());
			map.put("validationscores", results.get(2));
			map.put("runtime", results.get(3).asInt());
			processor.processResults(map);
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	public static JsonNode getPythonExperimentResults(final String options) throws InterruptedException, IOException, ProcessIDNotRetrievableException {
		File workingDirectory = new File("python/singularity");
		String id = UUID.randomUUID().toString();
		File folder = new File(workingDirectory.getAbsolutePath() + File.separator + "tmp/" + id);
		folder.mkdirs();

		File file = new File("runexperiment.py");
		String singularityImage = "test.simg";
		List<String> cmdList = Arrays.asList("singularity", "exec", singularityImage, "bash", "-c", "python3.8 " + file + " " + options + " --folder=" + folder.getAbsolutePath());
		logger.info("Executing {}", cmdList);

		ProcessBuilder pb = new ProcessBuilder(cmdList);
		pb.directory(workingDirectory);
		pb.redirectErrorStream(true);
		System.out.println("Starting process. Current memory usage is " + ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024) + "MB");
		Process p = pb.start();
		System.out.println("PID: " + ProcessUtil.getPID(p));
		try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
			String line;
			while ((line = br.readLine()) != null) {
				System.out.println(" --> " + line);
			}

			System.out.println("awaiting termination");
			while (p.isAlive()) {
				Thread.sleep(1000);
			}
			System.out.println("ready");

			return new ObjectMapper().readTree(FileUtil.readFileAsString(new File(folder + File.separator + "results.txt")));
		}
		finally {
			System.out.println("KILLING PROCESS!");
			ProcessUtil.killProcess(p);
		}
	}

	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, InterruptedException, ExperimentEvaluationFailedException, ExperimentFailurePredictionException {

		String databaseconf = args[0];
		String jobInfo = args[1];

		/* setup experimenter frontend */
		ExperimenterFrontend fe = new ExperimenterFrontend().withEvaluator(new ExperimentRunner()).withExperimentsConfig(new File("conf/experiments.conf")).withDatabaseConfig(new File(databaseconf));
		fe.setLoggerName("frontend");
		fe.withExecutorInfo(jobInfo);


		logger.info("Conducting experiment. Currently used memory is {}MB. Free memory is {}MB.", (Runtime.getRuntime().maxMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024.0), Runtime.getRuntime().freeMemory() / (1024 * 1024.0));
		fe.randomlyConductExperiments(1);
		logger.info("Experiment finished, stopping!");
	}
}
