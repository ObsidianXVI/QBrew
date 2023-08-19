part of qbrew;

QLAgent epsilonAgent1(
  Environment env,
  Logger logger,
) =>
    QLAgent(
      env: env,
      qTable: {},
      logger: logger,
      actionSelectionPolicy: ActionSelectionPolicy.softMax,
      epsilon: 0.8,
      gamma: 0.9,
      alpha: 0.2,
    );

QLAgent softmaxAgent1(
  Environment env,
  Logger logger,
) =>
    QLAgent(
      env: env,
      qTable: {},
      logger: logger,
      actionSelectionPolicy: ActionSelectionPolicy.softMax,
      epsilon: 0,
      gamma: 0.9,
      alpha: 0.2,
    );
