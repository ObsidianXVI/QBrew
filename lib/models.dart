part of qbrew;

typedef CustomerCountFunction = int Function(double currentPrice);

/// f(x) = 16 - 2x
int linear_1(double currentPrice) => (16 - 2 * currentPrice).round();

/// f(x) = 8 - 1.4x
int linear_2(double currentPrice) => (8 - 1.4 * currentPrice).round();

/// f(x) = -0.09x^2 + 1.1x + 3
int quadratic_1(double currentPrice) =>
    (-0.09 * pow(currentPrice, 2) + 1.1 * currentPrice + 16).round();

/// f(x) = -0.02x^3 + 16
int cubic_1(double currentPrice) => (-0.02 * pow(currentPrice, 3) + 16).round();
