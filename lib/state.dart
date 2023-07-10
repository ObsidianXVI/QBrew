part of qbrew;

class State {
  final double currentPrice;
  final int customers;
  List<Action> get actionsAvailable {
    if (currentPrice < 0) {
      return [
        Action(priceChange: 2),
        Action(priceChange: 4),
        Action(priceChange: 6),
      ];
    } else {
      return [
        Action(priceChange: -2),
        Action(priceChange: -4),
        Action(priceChange: -6),
        Action(priceChange: 2),
        Action(priceChange: 4),
        Action(priceChange: 6),
      ];
    }
  }

  const State({
    required this.currentPrice,
    required this.customers,
  });
}
