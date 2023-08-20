part of qbrew;

/// Run the same test multiple times and collate the data in one place.
/// Work with datasets as well.
class Dataset {
  final String label;
  final List<Map<String, List>> dataBlocks = [];
  final String dirPath;

  Dataset({
    required this.label,
  }) : dirPath = "./data/archives/datasets/$label";

  Future<void> batchRun({
    required int count,
    required Environment Function() createEnv,
    required QLAgent Function(Environment, Logger) createAgent,
    required Logger Function() createLogger,
  }) async {
    Environment? env;
    Logger? logger;
    QLAgent? agent;
    for (int i = 0; i < count; i++) {
      env = createEnv();
      logger = createLogger();
      agent = createAgent(env, logger);
      agent.start();
      dataBlocks.add(await logger.exportDataCSV('data_$i', dirPath));
    }
    if (count > 0) {
      await Logger.exportConfigsFile(
        'configs',
        directory: dirPath,
        env: env!,
        agent: agent!,
      );
    }
  }

  Future<void> exportRangeCSV() async {
    final Map<String, List> rangeCSV = {
      'timeStep': [],
      'rmax': [],
      'rmin': [],
      'ravg': [],
    };

    for (int ts = 0;
        ts < ((dataBlocks.first['timeStep'] as List)).length;
        ts++) {
      rangeCSV['timeStep']!.add(ts);
      final List<double> rewardsAtTimestep = [
        for (Map<String, List> block in dataBlocks) block['reward']![ts]
      ];

      rangeCSV['rmax']!.add(rewardsAtTimestep.reduce(max));
      rangeCSV['rmin']!.add(rewardsAtTimestep.reduce(min));
      rangeCSV['ravg']!.add(
          rewardsAtTimestep.reduce((a, b) => a + b) / rewardsAtTimestep.length);
    }

    final File dataFile = await File("$dirPath/range.csv").create();
    final List<String> headers = rangeCSV.keys.toList();
    final int rowCount = rangeCSV['timeStep']!.length;
    final List<String> data = [];
    data.add(headers.join(';'));
    for (int i = 0; i < rowCount; i++) {
      final List rowData = [];
      for (String header in headers) {
        rowData.add(rangeCSV[header]![i]);
      }
      data.add(rowData.join(';'));
    }
    dataFile.writeAsString(data.join('\n'));
  }
}
