#include <iostream>
#include <vector>
#include <chrono>

#define batchSize 100
#define numInputs 5
#define numOutputs 2
#define learningRate 0.01

using std::cout;
using std::endl;
using std::find;
using std::vector;
using std::distance;

using std::chrono::seconds;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

uint32_t m_z = (uint32_t)duration_cast<seconds>(high_resolution_clock::now().time_since_epoch()).count();
uint32_t m_w = (uint32_t)duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();

uint32_t intRand()
{
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);

	return (m_z << 16) + m_w;
}

double doubleRand() { return (intRand() + 1.0) * 2.328306435454494e-10; }

double activation(double x)
{
	x = exp(2 * x);
	return (x - 1) / (x + 1);
}

double dactivation(double x)
{
	x = exp(x) + exp(-x);
	return 4 / (x * x);
}

class DNN
{
public:
	vector<int> order;
	vector<bool> empty;
	vector<double> bias;
	vector<double> dbias;
	vector<double> output;
	vector<double> doutput;
	vector<double> preoutput;
	vector<vector<double>> weights;
	vector<vector<double>> dweights;
	vector<vector<int>> connections;

	DNN()
	{
		for (int i = 0; i < numInputs + numOutputs; i++)
		{
			dbias.push_back(0);
			output.push_back(0);
			doutput.push_back(0);
			preoutput.push_back(0);
			empty.push_back(false);
			bias.push_back(doubleRand() * 2 - 1);
		}

		for (int i = 0; i < numInputs + numOutputs; i++)
		{
			weights.push_back({});
			dweights.push_back({});
			connections.push_back({});

			if (i >= numInputs)
				for (int j = 0; j < numInputs; j++)
				{
					dweights[i].push_back(0);
					connections[i].push_back(j);
					weights[i].push_back(doubleRand() * 2 - 1);
				}
		}

		updateOrder();
	}

	void print()
	{
		cout << "output: ";

		for (int n = 0; n < output.size(); n++)
		{
			cout << output[n] << " ";
		}

		cout << endl << endl;

		cout << "doutput: ";

		for (int n = 0; n < doutput.size(); n++)
		{
			cout << doutput[n] << " ";
		}

		cout << endl << endl;

		cout << "preoutput: ";

		for (int n = 0; n < preoutput.size(); n++)
		{
			cout << preoutput[n] << " ";
		}

		cout << endl << endl;

		cout << "empty: ";

		for (int n = 0; n < empty.size(); n++)
		{
			cout << empty[n] << " ";
		}

		cout << endl << endl;

		cout << "bias: ";

		for (int n = 0; n < bias.size(); n++)
		{
			cout << bias[n] << " ";
		}

		cout << endl << endl;

		cout << "dbias: ";

		for (int n = 0; n < dbias.size(); n++)
		{
			cout << dbias[n] << " ";
		}

		cout << endl << endl;

		cout << "weights: " << endl;

		for (int n = 0; n < weights.size(); n++)
		{
			for (int i = 0; i < weights[n].size(); i++)
			{
				cout << weights[n][i] << " ";
			}

			cout << endl;
		}

		cout << endl;

		cout << "dweights: " << endl;

		for (int n = 0; n < dweights.size(); n++)
		{
			for (int i = 0; i < dweights[n].size(); i++)
			{
				cout << dweights[n][i] << " ";
			}

			cout << endl;
		}

		cout << endl;

		cout << "connections: " << endl;

		for (int n = 0; n < connections.size(); n++)
		{
			for (int i = 0; i < connections[n].size(); i++)
			{
				cout << connections[n][i] << " ";
			}

			cout << endl;
		}

		cout << endl;
	}

	void getNodeOrder(int node, bool* searched)
	{
		for (int c = 0; c < connections[node].size(); c++)
			if (!searched[connections[node][c]])
				getNodeOrder(connections[node][c], searched);/**/

		searched[node] = true;
		order.push_back(node);
	}

	void updateOrder()
	{
		order.clear();

		bool* searched = new bool[bias.size()]{};

		for (int i = 0; i < numInputs; i++) searched[i] = true;

		for (int o = numInputs; o < numInputs + numOutputs; o++)
			getNodeOrder(o, searched);

		delete[] searched;
	}

	void getNodeValue(int node, bool* searched)
	{
		for (int c = 0; c < connections[node].size(); c++)
		{
			int childNode = connections[node][c];

			if (!searched[childNode]) getNodeValue(childNode, searched);
			preoutput[node] += output[childNode] * weights[node][c];
		}

		output[node] = activation(preoutput[node]);
		searched[node] = true;
	}

	void forwardPropagate(double* inputs)
	{
		bool* searched = new bool[bias.size()]{};

		for (int i = 0; i < bias.size(); i++) preoutput[i] = bias[i];

		for (int i = 0; i < numInputs; i++)
		{
			searched[i] = true;
			output[i] = inputs[i];
		}

		for (int o = numInputs; o < numInputs + numOutputs; o++) getNodeValue(o, searched);

		delete[] searched;
	}

	void getNodeChange(int node)
	{
		double bBais = doutput[node] * dactivation(preoutput[node]);
		dbias[node] += bBais;

		for (int c = 0; c < connections[node].size(); c++)
		{
			int childNode = connections[node][c];

			dweights[node][c] += bBais * output[childNode];
			doutput[childNode] += bBais * weights[node][c];
		}
	}

	void backPropagate(double* inputs, double* outputs)
	{
		forwardPropagate(inputs);

		for (int o = numInputs; o < numInputs + numOutputs; o++)
			doutput[o] = output[o] - outputs[o - numInputs];

		for (int i = bias.size() - numInputs - 1; i >= 0; i--) getNodeChange(order[i]);
	}

	void applyPropagationChange(double inverseBatchSize)
	{
		for (int n = 0; n < bias.size(); n++)
		{
			bias[n] -= dbias[n] * learningRate * inverseBatchSize;
			dbias[n] = 0;

			for (int c = 0; c < weights[n].size(); c++)
			{
				weights[n][c] -= dweights[n][c] * learningRate * inverseBatchSize;
				dweights[n][c] = 0;
			}
		}
	}

	void addNode()
	{
		int node1 = doubleRand() * (bias.size() - numInputs) + numInputs;
		int connection = doubleRand() * connections[node1].size();
		vector<bool>::iterator findNode2 = find(empty.begin(), empty.end(), true);

		if (findNode2 == empty.end())
		{
			empty.push_back(false);
			bias.push_back(doubleRand() * 2 - 1);
			dbias.push_back(0);
			output.push_back(0);
			doutput.push_back(0);
			preoutput.push_back(0);
			weights.push_back({ doubleRand() * 2 - 1 });
			dweights.push_back({ 0 });
			connections.push_back({ connections[node1][connection] });
			connections[node1][connection] = bias.size() - 1;
		}
		else
		{
			int node2 = distance(empty.begin(), findNode2);

			empty[node2] = false;
			bias[node2] = doubleRand() * 2 - 1;
			dbias[node2] = 0;
			output[node2] = 0;
			doutput[node2] = 0;
			preoutput[node2] = 0;
			weights[node2] = { doubleRand() * 2 - 1 };
			dweights[node2] = { 0 };
			connections[node2] = { connections[node1][connection] };
			connections[node1][connection] = node2;
		}
	}

	void addConnection()
	{
		if (order.size() != numOutputs)
		{
			int node1 = doubleRand() * (order.size() - 1) + 1;
			int node2 = order[doubleRand() * node1];
			node1 = order[node1];

			vector<int>::iterator findNode2 = find(connections[node1].begin(), connections[node1].end(), node2);

			if (findNode2 == connections[node1].end())
			{
				weights[node1].push_back(doubleRand() * 2 - 1);
				dweights[node1].push_back(0);
				connections[node1].push_back(node2);
			}
		}
	}
};

int main()
{
	DNN agent;
	int tick = 0;

	/*agent.addNode();
	agent.updateOrder();*/
	/*for (int i = 0; i < 20; i++)
	{
		if (doubleRand() > 0.7)
		{
		}
		else {
			agent.addConnection();
			agent.updateOrder();
		}
	}*/

	while (true)
	{
		double error[numOutputs]{};

		for (int i = 0; i < batchSize; i++)
		{
			double sum = 0;
			double difference = 0;
			double input[numInputs]{};
			double output[numOutputs]{};

			for (int j = 0; j < numInputs; j++)
			{
				input[j] = doubleRand() * 2 - 1;
				output[0] += input[j];
				output[1] -= input[j];
			}

			agent.backPropagate(input, output);

			for (int o = 0; o < numOutputs; o++)
				error[o] += abs(agent.output[o + numInputs] - output[o]);
		}

		agent.applyPropagationChange(1.0 / batchSize);

		if (++tick == 10000)
		{
			tick = 0;

			cout << "error: ";

			for (int o = 0; o < numOutputs; o++)
				cout << error[o] / batchSize << " ";

			cout << endl;
		}
	}

	return 0;
}