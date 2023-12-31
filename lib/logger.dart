part of qbrew;

class Logger {
  final Map<String, dynamic Function(TimestepLog)> monitoredFeatures;
  final bool Function(TimestepLog) loggingCondition;
  final Map<String, List> csvData;
  final bool liveReporting;

  Logger({
    required this.monitoredFeatures,
    required this.loggingCondition,
    required this.liveReporting,
  }) : csvData = <String, List>{
          'timeStep': [],
          for (String key in monitoredFeatures.keys) key: [],
        } {
    init();
  }

  static const String delimiter = ';';

  void init() {
    monitoredFeatures.addAll({
      'timeStep': (TimestepLog tl) => tl.timestep,
    });
  }

  void logTimestep(TimestepLog timestepLog) {
    if (loggingCondition(timestepLog)) {
      for (String key in csvData.keys) {
        final dynamic accessValue = monitoredFeatures[key]!(timestepLog);
        csvData[key]!.add(accessValue);
      }
    }
    // if (liveReporting)
    //  print(
    //      '${env.timestep} | -> C:${env.customers} P:${env.drinkPrice} [${qValuesOfState[selectedQVector]}] (${optimalQVector.value})');
  }

  Future<Map<String, List>> exportDataCSV(String fileName,
      [String? directory]) async {
    directory ??= './data';
    await Directory(directory).create();
    final File dataFile =
        await File("$directory/$fileName.csv").create(recursive: true);
    final List<String> headers = csvData.keys.toList();
    final int rowCount = csvData['timeStep']!.length;
    final List<String> data = [];
    data.add(headers.join(delimiter));
    for (int i = 0; i < rowCount; i++) {
      final List rowData = [];
      for (String header in headers) {
        rowData.add(csvData[header]![i]);
      }
      data.add(rowData.join(delimiter));
    }
    dataFile.writeAsString(data.join('\n'));
    return csvData;
  }

  Future<void> exportFullTestArchive(
    String label, {
    String? directory,
    required Environment env,
    required QLAgent agent,
  }) async {
    directory ??= './data/archives/$label';
    await Directory(directory).create();
    await exportDataCSV('data', directory);
    await exportConfigsFile(label,
        directory: directory, env: env, agent: agent);
  }

  static Future<void> exportConfigsFile(
    String label, {
    String? directory,
    required Environment env,
    required QLAgent agent,
  }) async {
    final File configsFile =
        await File("$directory/configs.txt").create(recursive: true);
    configsFile.writeAsString("""Name: $label

############ Configuration
=== Execution
totalEpochs: $totalEpochs
epochSize: $epochSize
episodeSize: $episodeSize
epsilonDecayRate: $epsilonDecayRate

=== Agent
policy: ${agent.actionSelectionPolicy}
alpha: ${agent.alpha}
gamma: ${agent.gamma}
epsilon: ${agent.initialEpsilon}

=== Environment
Noise adjustments: ${env.noiseAdjustments}
Noisiness factor: ${env.noisinessFactor}

############ Results
=== Risk
CVaR: 

=== Responsiveness
TTC: 
""");
  }

  Future<String> dumpQTableCSV(
    String fileName,
    Map<QVector, double> qTable, [
    String? directory,
  ]) async {
    directory ??= './data';
    await Directory(directory).create();
    final File dataFile = await File("$directory/$fileName.csv").create();

    final List<String> headers = ['action', 'state', 'qvalue'];
    final int rowCount = qTable.length;
    final List<String> data = [];
    data.add(headers.join(delimiter));

    for (int i = 0; i < rowCount; i++) {
      final MapEntry<QVector, double> entry = qTable.entries.elementAt(i);
      data.add([
        entry.key.action.priceChange,
        entry.key.state,
        entry.value,
      ].join(delimiter));
    }

    dataFile.writeAsString(data.join('\n'));
    return dataFile.path;
  }
}

class TimestepLog {
  final int timestep;
  final State previousState;
  final Action chosenAction;
  final Action optimalAction;
  final double oldQValue;
  final double newQValue;
  final double reward;
  final bool rand;

  const TimestepLog({
    required this.timestep,
    required this.chosenAction,
    required this.newQValue,
    required this.oldQValue,
    required this.optimalAction,
    required this.previousState,
    required this.reward,
    required this.rand,
  });
}
