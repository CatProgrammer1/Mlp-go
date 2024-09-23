package mlpgo

import (
	"log"
	"math"
	"math/rand"
)

type MLP struct {
	learningRate float64
	layers       []Layer
}

func RandWeight() float64 {
	return -1 + rand.Float64()*(1-(-1))
}

func NewMLP(learningRate float64, weightsAmount int, neuronsAndLayersAmount []uint) *MLP {
	mlp := &MLP{
		learningRate: learningRate,
		layers:       make([]Layer, len(neuronsAndLayersAmount)),
	}

	for i := 0; i < len(neuronsAndLayersAmount); i++ {

		newLayer := Layer{}

		newNeuronsWeightsAmount := weightsAmount

		if len(mlp.layers)-1 >= 0 {
			newNeuronsWeightsAmount = len(mlp.layers[len(mlp.layers)-1].neurons)
		}

		for j := 0; j < int(neuronsAndLayersAmount[i]); j++ {

			newNeuron := Neuron{
				learningRate: learningRate,
				bias:         1,
			}

			for k := 0; k < newNeuronsWeightsAmount; k++ {
				newNeuron.weights = append(newNeuron.weights, RandWeight())
			}

			newLayer.neurons = append(newLayer.neurons, newNeuron)
		}

		mlp.layers[i] = newLayer
	}

	return mlp
}

func (mlp *MLP) Feed(input []float64) []float64 {
	var nextInput []float64 = input

	for i := 0; i < len(mlp.layers); i++ {
		output := mlp.layers[i].Feed(nextInput)
		nextInput = output
	}

	return nextInput
}

func (mlp *MLP) Train(input []float64, expected []float64) {
	var nextInput []float64 = input

	for i := 0; i < len(mlp.layers); i++ {
		mlp.layers[i].Train(nextInput, expected)
		nextInput = mlp.layers[i].Feed(nextInput)
	}
}

type Layer struct {
	neurons []Neuron
}

func (l *Layer) Feed(input []float64) []float64 {
	output := make([]float64, len(l.neurons))

	for i := 0; i < len(l.neurons); i++ {
		output[i] = l.neurons[i].Feed(input)
	}

	return output
}

func (l *Layer) Train(input []float64, expected []float64) {
	for i := 0; i < len(l.neurons); i++ {
		if len(l.neurons) != len(expected) {
			log.Fatalln(`Amount of neurons in the layer and length of "expected" array must be the same.
			Neurons amount:`, len(l.neurons), `;Lenght of "expected" array:`, len(expected))
		}
		l.neurons[i].Train(input, expected[i])
	}
}

type Neuron struct {
	learningRate float64
	bias         float64
	weights      []float64
}

func (n *Neuron) Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (n *Neuron) SigmoidDerative(x float64) float64 {
	return x * (1 - x)
}

func (n *Neuron) Feed(input []float64) float64 {
	var sum float64

	for i := 0; i < len(n.weights); i++ {
		sum += input[i] * n.weights[i]
	}

	sum += n.bias

	return n.Sigmoid(sum)
}

func (n *Neuron) Train(input []float64, expected float64) {
	result := n.Feed(input)

	err := expected - result

	for i := 0; i < len(n.weights); i++ {
		n.weights[i] += n.learningRate * err * n.SigmoidDerative(result) * input[i]
	}

	n.bias += n.learningRate * err * n.SigmoidDerative(result)
}
