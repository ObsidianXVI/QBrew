part of qbrew;

class Environment {
  final double noisinessFactor;
  final List<int> noiseAdjustments;
  final Random _random = Random();
  List<CustomerCountFunction> customerCountFunctions;
  CustomerCountFunction currentFn;
  double drinkPrice = 5;
  int customers = 10;
  int timestep = 0;

  Environment({
    required this.customerCountFunctions,
    required this.noiseAdjustments,
    required this.noisinessFactor,
  }) : currentFn = customerCountFunctions.first;

  double computeReward(Action action) {
    drinkPrice = drinkPrice + action.priceChange;
    customers = computeCustomerCount(drinkPrice);
    return customers * drinkPrice;
  }

  int computeCustomerCount(double currentPrice) {
    final int newCustomerCount = currentFn(currentPrice);
    final double randDbl = _random.nextDouble();
    if (randDbl > noisinessFactor) {
      return newCustomerCount;
    } else {
      return newCustomerCount +
          noiseAdjustments[_random.nextInt(noiseAdjustments.length)];
    }
  }
}
