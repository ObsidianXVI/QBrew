part of qbrew;

const double noisinessFactor = 0.4;
const List<int> noiseAdjustments = [-3, -2, -1, 1, 2, 3];

class Environment {
  final Random _random = Random();
  double drinkPrice = 5;
  int customers = 10;
  int timestep = 0;

  double computeReward(Action action) {
    drinkPrice = drinkPrice + action.priceChange;
    customers = computeCustomerCount(drinkPrice);
    return customers * drinkPrice;
  }

  // use new
  // f(x)=-0.01x^{2}+1.1x+3
  int computeCustomerCount(double currentPrice) {
    final int newCustomerCount = (16 - 2 * currentPrice).round();
    final double randDbl = _random.nextDouble();
    if (randDbl > noisinessFactor) {
      return newCustomerCount;
    } else {
      return newCustomerCount +
          noiseAdjustments[_random.nextInt(noiseAdjustments.length)];
    }
  }
}
