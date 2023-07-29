part of qbrew;

class QVector {
  final State state;
  final Action action;

  QVector({
    required this.action,
    required this.state,
  });

  @override
  String toString() => "QVect<$state, $action>";

  @override
  bool operator ==(Object? other) {
    return (other is QVector) && other.state == state && other.action == action;
  }
}
