part of qbrew;

class Action {
  final double priceChange;

  const Action({
    required this.priceChange,
  });

  @override
  String toString() {
    return "A<$priceChange>";
  }

  @override
  bool operator ==(Object? other) {
    return (other is Action) && other.priceChange == priceChange;
  }
}
