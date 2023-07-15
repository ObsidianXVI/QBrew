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
  final Logger logger = Logger(monitoredFeatures: {
    'prevState': (tl) => tl.previousState,
    'chosenAction': (tl) => tl.chosenAction,
    'oldQValue': (tl) => tl.oldQValue,
    'newQValue': (tl) => tl.newQValue,
    'rand': (tl) => tl.rand,
    'reward': (tl) => tl.reward,
  });

  final QLAgent agent = QLAgent(
    env: Environment(),
    qTable: {},
    logger: logger,
  );

  for (int j = 0; j < 50; j++) {
    State state = State(currentPrice: 5, customers: 10);
    for (int i = 0; i < 20; i++) {
      state = agent.perform(state);
    }
  }

  await logger.exportCSV('v001_1');
}

class QLAgent {
  final Map<QVector, double> qTable;
  final Environment env;
  final Logger? logger;
  final Random _random = Random();
  static const double epsilon = 0.2;
  static const double gamma = 0.9;
  static const double alpha = 0.2;

  QLAgent({
    required this.env,
    required this.qTable,
    required this.logger,
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
      if (policy == ActionSelectionPolicy.epsilonGreedy) {
        // epsilon cannot be const when it is decayed
        final double randNum = _random.nextDouble();
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
      } else {
        isRand = false;
        return optimalQVector.key;
      }
    }

    final QVector selectedQVector =
        selectAction(ActionSelectionPolicy.epsilonGreedy);
    final Action selectedAction = selectedQVector.action;

    // perform action
    final double reward = env.computeReward(selectedAction);
    print(
        '${env.timestep} | -> C:${env.customers} P:${env.drinkPrice} [${qValuesOfState[selectedQVector]}] (${optimalQVector.value})');
    env.timestep += 1;
    final State newState = State(
      currentPrice: env.drinkPrice,
      customers: env.customers,
    );
    final double maxFutureValue = computeMaxFutureQValue(newState);
    final double newQValue = reward + gamma * maxFutureValue;

    // calculate temporal difference
    final double temporalDifference = newQValue - optimalQVector.value;

    // compute new Q-value
    final double oldQValue = qValuesOfState[selectedQVector]!;
    final double updatedQValue =
        optimalQVector.value + alpha * temporalDifference;

    // update Q-value of chosen action and argset in Q-table
    print('          $updatedQValue');
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
      final QVector qv = QVector(action: action, state: state);
      if (qTable.containsKey(qv)) {
        qvalues[qv] = qTable[qv]!;
      } else {
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
