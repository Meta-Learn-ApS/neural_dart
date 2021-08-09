import 'package:neural_dart/neural.dart';

void main(List<String> arguments) async {
  Parser parser = Parser();
  Network network = await parser.parse();
  // network.show();
  List<double> y = network.calculate([0.0,-1.1,2.2,3.3,4.4,-5.5,6.6,7.7,8.8,9.9]);
  print(y);
}
