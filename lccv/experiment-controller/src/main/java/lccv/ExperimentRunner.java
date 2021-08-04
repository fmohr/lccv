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
			String algo = keys.get("algorithm");
			String timeout = keys.get("timeout");
			logger.info("\topenmlid: {}", openmlid);
			logger.info("\talgo: {}", algo);
			logger.info("\tseed: {}", seed);
			logger.info("\ttimeout: {}", timeout);

			/* run python experiment */
			String options = openmlid + " " + algo + " " + seed + " " + timeout;
			List<Object> results = getPythonExperimentResults(options);

			/* write results */
			Map<String, Object> map = new HashMap<>();
			map.put("chosenmodel", results.get(0));
			map.put("errorrate", results.get(1));
			map.put("runtime", results.get(2));
			processor.processResults(map);
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	public static List<Object> getPythonExperimentResults(final String options) throws InterruptedException, IOException, ProcessIDNotRetrievableException {
		File workingDirectory = new File("python/singularity");
		String id = UUID.randomUUID().toString();
		File folder = new File(workingDirectory.getAbsolutePath() + File.separator + "tmp/" + id);
		folder.mkdirs();

		File file = new File("runexperiment.py");
		String singularityImage = "test.simg";
		List<String> cmdList = Arrays.asList("singularity", "exec", singularityImage, "bash", "-c", "python3 " + file + " " + options + " " + folder.getAbsolutePath());
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

			String[] results = FileUtil.readFileAsString(new File(folder + File.separator + "results.txt")).split(" ");
			return Arrays.asList(results[0], results[1], results[2]);
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
