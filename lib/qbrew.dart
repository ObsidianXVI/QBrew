library qbrew;

import 'dart:io';
import 'dart:math';

part './state.dart';
part './qvector.dart';
part './action.dart';
part './environment.dart';
part './logger.dart';

enum ActionSelectionPolicy {
  epsilonGreedy,
  softMax,
}

void main(List<String> args) async {
  const int totalEpochs = 1000; // 500
  const int totalEpisodes = 40; // 40
  const int totalTimesteps = totalEpisodes * totalEpochs;

  final Logger logger = Logger(
      liveReporting: false,
      monitoredFeatures: {
        'prevPrice': (tl) => tl.previousState.currentPrice,
        'prevCustomers': (tl) => tl.previousState.customers,
        'chosenAction': (tl) => tl.chosenAction.priceChange,
        'oldQValue': (tl) => tl.oldQValue,
        'newQValue': (tl) => tl.newQValue,
        'rand': (tl) => tl.rand,
        'reward': (tl) => tl.reward,
      },
      // only log for the final 10% of timesteps
      loggingCondition: (tl) => true //tl.timestep > (0.9 * totalTimesteps),
      );

  final QLAgent agent = QLAgent(
    env: Environment(),
    qTable: {},
    logger: logger,
    actionSelectionPolicy: ActionSelectionPolicy.softMax,
  );

  final State initialState = State(currentPrice: 5, customers: 10);
  for (int epoch = 0; epoch < totalEpochs; epoch++) {
    State state = initialState;
    for (int epiode = 0; epiode < totalEpisodes; epiode++) {
      state = agent.perform(state);
      // Decay the epsilon value
      QLAgent.epsilon -= 0.00001;
    }
  }

  await logger.exportDataCSV('sm_g0_v0');
  // await logger.dumpQTableCSV('qtable_v1', agent.qTable);
}

class QLAgent {
  final Map<QVector, double> qTable;
  final Environment env;
  final Logger? logger;
  final Random _random = Random();
  final ActionSelectionPolicy actionSelectionPolicy;
  static double epsilon = 0.6;
  static const double gamma = 0.9;
  static const double alpha = 0.2;

  QLAgent({
    required this.env,
    required this.qTable,
    required this.logger,
    required this.actionSelectionPolicy,
  });

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
