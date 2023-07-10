library qbrew;

import 'dart:math';

part './state.dart';
part './qvector.dart';
part './action.dart';
part './environment.dart';

void main(List<String> args) {
  final QLAgent agent = QLAgent(env: Environment(), qTable: {});
  for (int j = 0; j < 50; j++) {
    State state = State(currentPrice: 5, customers: 10);
    for (int i = 0; i < 20; i++) {
      state = agent.perform(state);
    }
  }
}

class QLAgent {
  final Map<QVector, double> qTable;
  final Environment env;

  QLAgent({
    required this.env,
    required this.qTable,
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

    // select action using policy
    final Action selectedAction = optimalQVector.key.action;

    // perform action
    final double reward = env.computeReward(selectedAction);
    print(
        '${env.timestep} | -> C:${env.customers} P:${env.drinkPrice} [${selectedAction.priceChange}] (${optimalQVector.value})');
    env.timestep += 1;
    final State newState = State(
      currentPrice: env.drinkPrice,
      customers: env.customers,
    );
    final double maxFutureValue = computeMaxFutureQValue(newState);
    final double newQValue = reward + 0.9 * maxFutureValue;

    // calculate temporal difference
    final double temporalDifference = newQValue - optimalQVector.value;

    // compute new Q-value
    final double updatedQValue =
        optimalQVector.value + 0.2 * temporalDifference;

    // update Q-value of chosen action and argset in Q-table
    print('          $updatedQValue');
    qTable[optimalQVector.key] = updatedQValue;
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
