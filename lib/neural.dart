import 'dart:async';
import 'dart:io';
import 'dart:convert';


extension BoolParsing on String {
  bool parseBool() {
    return this.toLowerCase() == 'true';
  }
}

class Layer {
  List<double> calculate(List<double> x) {
    return x;
  }
}

class Linear extends Layer {
  final int in_features;
  final int out_features;
  final List<List<double>> weight;
  final List<double>? bias;
  Linear(this.in_features, this.out_features, this.weight, this.bias);

  @override
  List<double> calculate(List<double> x) {
    List<double> ax = weight.map((element) => [for (int i=0; i<element.length; i+=1) element[i] * x[i]].reduce((a, b) => a + b)).toList();
    if (bias != null) return [for (int i=0; i<ax.length; i+=1) ax[i] + bias![i]];
    return ax;
  }
}

class ReLU extends Layer {
  @override
  List<double> calculate(List<double> x) {
    return x.map((double val) => val <= 0 ? 0.0 : val).toList();
  }
}

class Network {
  List<Layer> layers;
  Network(this.layers);

  void show() {
    for (Layer layer in layers) {
      print(layer);
      if (layer is Linear) {
        print(layer.weight);
        print(layer.bias);
      }
    }
  }

  List<double> calculate(List<double> x) {
    for (Layer layer in layers) {
      x = layer.calculate(x);
    }
    return x;
  }
}

class Parser {
  final String file;
  const Parser(this.file);
  
  Future<Network> parse() async {
    Map<String, dynamic> file = jsonDecode(await get_file());
    Network network = handle_model(List<String>.from(file['model']), file['parameters']);
    return network;
  }




  Network handle_model(List<String> model, List<dynamic> parameters) {
    return Network(model.map((String description) => layer(description, parameters)).toList());
  }

  Layer layer(String description, List<dynamic> parameters) {
    String name = description.split("(")[0];
    List<String> values = ((description.split("(")[1].split("")..removeLast()).join()).split(',');
    switch (name) {
      case "Linear": return Linear(int.parse(values[0]), int.parse(values[1]), List<List<double>>.from(parameters.removeAt(0).map((e) => List<double>.from(e)).toList()), (values[2].parseBool()) ? List<double>.from(parameters.removeAt(0)) : null);
      case "ReLU":  return ReLU();
      default: return Layer();
    }
  }

  Future<String> get_file() {
    return File(name).readAsString().then((String contents) {
      return contents;
    });
  }
}