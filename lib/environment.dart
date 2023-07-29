part of qbrew;

class Environment {
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
  int computeCustomerCount(double currentPrice) =>
      (16 - 2 * currentPrice).round();
}
