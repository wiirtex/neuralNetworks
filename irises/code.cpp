// Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

typedef double dd;

using namespace std;

class Matrix {
public:
    int n; int m;
    vector<vector<dd>> data;

    Matrix() {
        data.resize(n);
        for (int i = 0; i < n; i++) {
            data[i].resize(m);
        }
    }

    Matrix(int n_, int m_) {
        n = n_;
        m = m_;
        data.resize(n);
        for (int i = 0; i < n; i++) {
            data[i].resize(m);
        }
    }

    void Transpose() {
        vector<vector<dd>> newdata(m);
        for (int i = 0; i < n; i++) {
            newdata[i].resize(n);
            for (int j = 0; j < m; j++)
            {
                newdata[j][i] = data[i][j];
            }
        }
        swap(m, n);
        data = newdata;
    }

    void Print(const char* sep = " ", const char* lineend = "\n") {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                cout << data[i][j] << sep;
            }
            cout << lineend;
        }
    }

    template <int k, int p>
    friend Matrix operator* (Matrix& a, const Matrix& b);
};


Matrix operator*(Matrix& a, const Matrix& b)
{
    Matrix res;
    
    for (int i = 0; i < a.n; i++) {
        for (int k = 0; k < b.m; k++) {
            for (int j = 0; j < b.m; j++) {
                res.data[i][k] += a.data[i][j] * b.data[j][k];
            }
        }
    }
    return res;
}

dd coefficient = 1;
dd speed_coefficient = 1;

inline dd normalize(dd x) {
    return 1. / (1 + pow(2.7182818284, -x));
}

class Layer : public Matrix {
public:
    int n; int next_layer_size = 0;
    Layer(int n_, int next_layer_size_ = 0) : Matrix(n_, 1) {
        n = n_;
        next_layer_size = next_layer_size_;
        sigmas.resize(n);
        outs.resize(n);
        weights.resize(n);
        for (int i = 0; i < n; i++) {
            weights[i].resize(next_layer_size);
            for (int j = 0; j < next_layer_size; j++) {
                weights[i][j] = rand() % 999983 / 999983.;
            }
        }
    }

    void SetWeights(vector<vector<dd>> weights_) {
        if (weights.size() != weights_.size()) {
            throw length_error("sizes are not equal");
        }
        weights = weights_;
    }
    vector<dd> sigmas;
    vector<vector<dd>> weights;
    vector<dd> outs;
    void PrintWeights(const char* sep = " ", const char* lineend = "\n") {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < next_layer_size; j++) {
                cout << weights[i][j] << sep;
            }
            cout << lineend;
        }
    }

    void PrintOuts(const char* sep = " ", const char* lineend = "\n") {
        for (int i = 0; i < n; i++) {
            cout << outs[i] << sep;
            cout << lineend;
        }
    }

    void PassDataToNextLayer(Layer& next_layer) {
        for (int i = 0; i < next_layer_size; i++) {
            for (int j = 0; j < n; j++) {
                next_layer.outs[i] += this->outs[j] * this->weights[j][i];
            }
            next_layer.outs[i] = normalize(next_layer.outs[i]);
        }
    }

    void CalculateSigmas(Layer& next_layer) {
        for (int i = 0; i < n; i++) {
            dd s = 0;
            for (int j = 0; j < next_layer_size; j++) {
                s += next_layer.sigmas[j] * this->weights[i][j];
            }
            this->sigmas[i] = this->outs[i] * (1 - this->outs[i]) * s;
        }
    }

    void CalculateSigmas(vector<dd> test_answers) {
        for (int i = 0; i < n; i++) {
            this->sigmas[i] = -this->outs[i] * (1 - this->outs[i]) * (test_answers[i] - this->outs[i]);
        }
    }

    void CorrectWeights(Layer& next_layer) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < next_layer_size; j++) {
                dd dw = speed_coefficient * this->outs[i] * next_layer.sigmas[j];
                this->weights[i][j] -= dw;
            }
        }
    }

    void Fill(vector<dd> data_) {
        if (outs.size() != data_.size()) {
            return;
        }
        outs = data_;
    }

};

// rand() % 999983 / 999983.

vector<vector<dd>> inputs;
vector<vector<dd>> outputs;
vector<vector<dd>> check_inputs;
vector<vector<dd>> check_outputs;
int epochs = 1500;


void train() {
    //srand(10000);
    Layer input(4, 50);
    Layer hidden(50, 3);
    Layer output(3);
    for (int e = 0; e < epochs; e++) {
        cout << endl << "Epoch number: " << e;
        dd error = 0;
        for (int i = 0; i < inputs.size(); i++) {
            input.Fill(inputs[i]);
            input.PassDataToNextLayer(hidden);
            hidden.PassDataToNextLayer(output);
            output.CalculateSigmas(outputs[i]);
            hidden.CalculateSigmas(output);
            input.CalculateSigmas(hidden);
            input.CorrectWeights(hidden);
            hidden.CorrectWeights(output);
            error += abs(outputs[i][0] - output.outs[0]);
        }
        cout << "   Error: " << error;
    }
    cout << endl << "Input Weights: {";
    input.PrintWeights(", ", "}, {");
    cout << "}" << endl;
    cout << endl << "Hidden Weights: {";
    hidden.PrintWeights(", ", "}, {");
    cout << "}" << endl;
}

void show() {
    dd coefficient = 0.1;
    //srand(10000);
    Layer input(4, 50);
    Layer hidden(50, 3);
    Layer output(3);
    input.SetWeights({ {4.729411, 0.347840, 2.400575, 3.401949, 0.412361, 3.271184, 1.230617, 0.734010, -0.019715, -2.835637, 3.891036, -0.323743, 0.142259, 0.342234, 3.501289, 4.405945, 0.339213, 0.872506, 2.318479, 0.327079, 2.548501, 0.365735, 4.743202, 0.820863, 3.980226, 0.720280, 0.357302, 3.216960, 2.138023, 0.905644, 1.756670, 3.644731, 2.580936, 2.118827, 0.354745, 3.720773, -1.543061, 0.417676, 1.777220, 4.512228, 3.083197, 3.425447, -0.142075, 2.125323, 0.889416, 4.113952, 0.202218, 4.395736, -4.134377, 4.451511, }, {3.500029, 1.420393, 5.463085, 1.990731, 1.712967, 4.922698, 1.345483, 0.890149, -0.055158, -3.574784, 3.927407, -1.036974, 0.827686, 1.356267, 2.451342, 3.233256, 1.338439, 1.005314, 6.109747, 1.304737, 2.245149, 1.617914, 2.581952, 0.917844, 5.330257, 0.934942, 1.394034, 2.255806, 6.261386, 0.947253, 1.713206, 3.574529, 2.459114, 2.031343, 1.403924, 2.155516, -1.089415, 1.147222, 1.655738, 3.991335, 3.818802, 2.808954, -0.012273, 1.369132, 0.943229, 2.990312, 1.187476, 2.749819, -4.904962, 1.597930, }, {-5.756014, -2.561935, -4.030955, -3.895230, -2.936748, -4.742629, -1.826647, -1.442860, 0.464937, 3.740753, -5.011524, 0.890691, -0.890670, -2.498767, -4.159205, -5.302666, -2.458459, -1.518064, -4.294511, -2.403112, -3.116077, -2.796466, -5.497163, -1.465366, -5.203082, -1.421749, -2.544296, -3.767967, -4.179409, -1.534820, -2.271117, -4.602964, -3.382122, -2.733001, -2.548743, -4.299090, 0.290519, -1.440075, -2.298335, -5.667151, -4.287349, -4.197221, 0.544571, -2.487118, -1.522864, -4.950946, -2.186709, -5.190402, 5.287125, -4.881677, }, {-9.084226, -1.106647, -8.552103, -6.927140, -1.244499, -9.003112, -3.990115, -2.458857, 0.860674, 4.473398, -8.815479, 2.063851, -1.708356, -1.083953, -7.367344, -8.761134, -1.075384, -2.976930, -8.194873, -1.051937, -6.500617, -1.161938, -8.445652, -2.775580, -8.874834, -2.548489, -1.105507, -7.105171, -8.332161, -2.950042, -5.072455, -8.522307, -6.032645, -5.693189, -1.105391, -7.331695, 0.759882, -1.673795, -5.045409, -9.329058, -7.577916, -7.540771, 0.798670, -5.139745, -2.916287, -8.254572, -0.906368, -8.240741, 6.097061, -7.515697, }, });
    hidden.SetWeights({ {-0.087315, 1.137507, -1.028513, }, {0.750804, -3.440264, -0.961082, }, {-0.116959, 0.735061, -0.462553, }, {-0.058895, 0.943277, -0.969617, }, {0.765325, -4.928491, -0.525973, }, {-0.078115, 0.742049, -0.763577, }, {0.401345, 0.771086, -2.166516, }, {0.571084, 0.103830, -2.485878, }, {-3.135484, -1.930894, 1.082068, }, {-1.306388, -0.698020, 0.179724, }, {-0.094452, 0.949855, -0.983067, }, {-3.555146, -0.518421, 1.588629, }, {0.043982, 0.324285, -1.724650, }, {0.750795, -3.206871, -0.919696, }, {-0.056497, 1.084524, -1.006048, }, {-0.029951, 1.081053, -1.058744, }, {0.891933, -3.112750, -1.163497, }, {0.504761, 0.370820, -2.454474, }, {0.138956, 0.386452, -0.478595, }, {0.880641, -2.955088, -1.225313, }, {0.024190, 1.163212, -0.908619, }, {0.615012, -4.251076, -0.378380, }, {-0.075024, 1.224376, -1.151791, }, {0.535177, 0.263334, -2.462019, }, {-0.040638, 0.887856, -0.873202, }, {0.547149, 0.147854, -2.465956, }, {0.891792, -3.373770, -1.084440, }, {-0.051201, 1.055619, -0.976708, }, {0.044348, 0.630894, -0.400643, }, {0.502316, 0.365920, -2.456335, }, {0.240698, 0.826356, -1.206386, }, {0.140765, 1.071408, -1.036453, }, {0.063114, 1.003966, -0.694522, }, {0.160714, 0.972615, -1.083183, }, {0.904525, -3.393013, -1.094377, }, {-0.060642, 1.097772, -1.156615, }, {-2.865786, -0.709229, 1.926180, }, {0.677015, -0.415248, -2.458908, }, {0.225046, 0.904735, -1.399057, }, {-0.094695, 1.093778, -0.957517, }, {-0.122557, 0.816163, -0.636033, }, {-0.125174, 0.986574, -0.955979, }, {-2.578335, -1.473303, 0.721009, }, {0.224402, 0.733412, -1.037934, }, {0.516609, 0.344887, -2.462815, }, {-0.082603, 1.008298, -1.096435, }, {0.241170, -2.337268, -0.462007, }, {-0.124590, 1.130401, -1.083846, }, {-1.342154, -0.897366, 1.011559, }, {-0.062549, 1.125697, -1.281856, }, });
    for (int i = 0; i < check_inputs.size(); i++) {
        input.Fill(check_inputs[i]);
        cout << "Input values: ";
        for (int q = 0; q < check_inputs[i].size(); q++) {
            cout << check_inputs[i][q] << ", ";
        }
        input.PassDataToNextLayer(hidden);
        hidden.PassDataToNextLayer(output);
        
        cout << "   Output values: ";
        output.PrintOuts(", ", "");
        cout << endl;
    }
}

int main()
{
    cout << fixed << setw(8);

    ifstream in;
    in.open("D:/Projects/iris.txt");
    for (int i = 0; i < 135; i++) {
        dd a, b, c, d;
        in >> a >> b >> c >> d;
        inputs.push_back({ a, b, c, d });
        dd e, f, g;
        in >> e >> f >> g;
        outputs.push_back({ e, f, g });
    }

    ifstream nin;
    nin.open("D:/Projects/check_iris.txt");
    for (int i = 0; i < 15; i++) {
        dd a, b, c, d;
        nin >> a >> b >> c >> d;
        check_inputs.push_back({ a, b, c, d });
        dd e, f, g;
        nin >> e >> f >> g;
        check_outputs.push_back({ e, f, g });
    }
    show();
    //show();
}


// 9.67335, -0.208361, -4.6295

// 0.350348, 2.10666, 9.33999,

// {3.124413, -1.860961, 0.963844, }, {0.387853, -1.242675, 4.371292, }, {-3.908398, -0.410930, -7.964784, }, {2.563295, 0.112786, -4.295957, }
// {-6.841622, -1.987036, 0.964185, }, {-4.934207, -2.175299, 1.325087, }, {12.392973, -4.362565, -8.683201, },
