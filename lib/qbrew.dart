library qbrew;

import 'dart:io';
import 'dart:math';

part './state.dart';
part './qvector.dart';
part './action.dart';
part './environment.dart';
part './models.dart';
part './logger.dart';
part './agents.dart';
part './dataset.dart';

enum ActionSelectionPolicy {
  epsilonGreedy,
  softMax,
}

const int totalEpochs = 4;
const int epochSize = 1000; // 1000
const int episodeSize = 40; // 40
const int totalTimesteps = episodeSize * epochSize * totalEpochs;
const double epsilonDecayRate = -0.000001;

void main(List<String> args) async {
  print(totalTimesteps);

  final Dataset dataset1 = Dataset(
    label: 'EG_D1',
  );

  await dataset1.batchRun(
    count: 3,
    createEnv: () => Environment(
      customerCountFunctions: [
        linear_1,
        quadratic_1,
        linear_2,
        cubic_1,
      ],
      noiseAdjustments: [-3, -2, -1, 1, 2, 3],
      noisinessFactor: 0.8,
    ),
    createAgent: (Environment env, Logger logger) => epsilonAgent1(env, logger),
    createLogger: () => Logger(
      liveReporting: false,
      monitoredFeatures: {'reward': (tl) => tl.reward},
      loggingCondition: (tl) => true,
    ),
  );
  await dataset1.exportRangeCSV();

  // await logger.exportFullTestArchive('SM_D1_3', env: env, agent: agent);
}

class QLAgent {
  final Map<QVector, double> qTable;
  final Environment env;
  final Logger? logger;
  final Random _random = Random();
  final ActionSelectionPolicy actionSelectionPolicy;
  final double initialEpsilon;
  double epsilon;
  final double gamma;
  final double alpha;

  QLAgent({
    required this.env,
    required this.qTable,
    required this.logger,
    required this.actionSelectionPolicy,
    required this.epsilon,
    required this.gamma,
    required this.alpha,
  }) : initialEpsilon = epsilon;

  void start() {
    final State initialState = State(currentPrice: 5, customers: 10);
    for (int epoch = 0; epoch < totalEpochs; epoch++) {
      env.currentFn = env.customerCountFunctions[epoch];
      for (int episode = 0; episode < epochSize; episode++) {
        State state = initialState;
        for (int timestep = 0; timestep < episodeSize; timestep++) {
          state = perform(state);
          // Decay the epsilon value
          if (epsilon - epsilonDecayRate >= 0) {
            epsilon -= epsilonDecayRate;
          } else {
            epsilon = 0;
          }
        }
      }
    }
  }

  State perform(State state) {
    // fetch available  Q-values
    final Map<QVector, double> qValuesOfState = fetchHistoricalQValues(state);

    // find optimal action
    MapEntry<QVector, double> optimalQVector = qValuesOfState.entries.first;
    for (MapEntry<QVector, double> qVector in qValuesOfState.entries) {
      if (qVector.value > optimalQVector.value) {
        optimalQVector = qVector;
      }
    }

    bool? isRand;
    // select action using policy
    QVector selectAction(ActionSelectionPolicy policy) {
      final double randNum = _random.nextDouble();

      if (policy == ActionSelectionPolicy.epsilonGreedy) {
        // epsilon cannot be const when it is decayed
        if (randNum < epsilon) {
          isRand = true;
          // find and perform a random action ("exploration")
          final int randIndex = _random.nextInt(qValuesOfState.length);
          return qValuesOfState.keys.elementAt(randIndex);
        } else {
          isRand = false;
          // perform the action currently known to be optimal ("exploitation")
          return optimalQVector.key;
        }
      } else if (actionSelectionPolicy == ActionSelectionPolicy.softMax) {
        isRand = false;
        final Map<QVector, Range> probabilityDistribution = {};
        double sumOfExponentials =
            qValuesOfState.values.map((e) => exp(e)).fold(0, (a, b) => a + b);
        double lastVal = -0.1;
        for (MapEntry<QVector, double> entry in qValuesOfState.entries) {
          final double prob = exp(entry.value) / sumOfExponentials;
          final double newVal = lastVal + prob;
          probabilityDistribution[entry.key] = Range(lastVal + 0.1, newVal);
          lastVal = newVal;
        }
        for (MapEntry<QVector, Range> distribution
            in probabilityDistribution.entries) {
          if (distribution.value.contains(randNum)) {
            return distribution.key;
          }
        }
        return optimalQVector.key;
      } else {
        return optimalQVector.key;
      }
    }

    final QVector selectedQVector = selectAction(actionSelectionPolicy);
    final Action selectedAction = selectedQVector.action;

    // perform action
    final double reward = env.computeReward(selectedAction);

    env.timestep += 1;

    final State newState = State(
      currentPrice: env.drinkPrice,
      customers: env.customers,
    );
    final double maxFutureValue = computeMaxFutureQValue(newState);
    // print('MFV: $maxFutureValue');
    final double newQValue = reward + gamma * maxFutureValue;
    // print('NQV: $newQValue');

    // calculate temporal difference
    final double temporalDifference = newQValue - optimalQVector.value;

    // compute new Q-value
    final double oldQValue = qValuesOfState[selectedQVector]!;
    final double updatedQValue =
        optimalQVector.value + alpha * temporalDifference;

    // update Q-value of chosen action and argset in Q-table
    qTable[optimalQVector.key] = updatedQValue;
    logger?.logTimestep(
      TimestepLog(
        timestep: env.timestep,
        chosenAction: selectedAction,
        newQValue: newQValue,
        oldQValue: oldQValue,
        optimalAction: optimalQVector.key.action,
        previousState: state,
        reward: reward,
        rand: isRand ?? false,
      ),
    );
    return newState;
  }

  Map<QVector, double> fetchHistoricalQValues(State state) {
    final Map<QVector, double> qvalues = {};
    for (Action action in state.actionsAvailable) {
      final MapEntry<QVector, double>? qVect = qTable.qVectorFor(action, state);
      if (qVect != null) {
        qvalues.addEntries([qVect]);
      } else {
        final QVector qv = QVector(action: action, state: state);
        qTable.addAll({qv: 0});
        qTable[qv] = 0;
        qvalues[qv] = 0;
      }
    }
    return qvalues;
  }

  double computeMaxFutureQValue(State newState) {
    final Map<QVector, double> qValuesOfState =
        fetchHistoricalQValues(newState);
    final double maxQValue = qValuesOfState.values.toList().reduce(max);
    return maxQValue;
  }
}

class Range {
  final double start;
  final double end;

  const Range(this.start, this.end);

  bool contains(double value) => start <= value && value <= end;
}

extension on Map<QVector, double> {
  MapEntry<QVector, double>? qVectorFor(Action a, State s) {
    for (MapEntry<QVector, double> entry in entries) {
      if (entry.key.action == a && entry.key.state == s) {
        return entry;
      }
    }
    return null;
  }
}
