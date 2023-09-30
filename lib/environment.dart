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

  /// Computes the new number of customers patronising the cafe at the given
  /// price of [currentPrice]
  int computeCustomerCount(double currentPrice) {
    // currentFn is the current demand function being used by env
    final int newCustomerCount = currentFn(currentPrice);
    // generate a random decimal between 0 and 1
    final double randDbl = _random.nextDouble();
    if (randDbl > noisinessFactor) {
      // return the exact new customer count if above the threshold
      return newCustomerCount;
    } else {
      // add random noise and return a slightly inaccurate count
      return newCustomerCount +
          noiseAdjustments[_random.nextInt(noiseAdjustments.length)];
    }
  }
}
