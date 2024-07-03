g++ testBasicOperators.cpp ../matrix/naive.cpp -o basicNaive
g++ testExp.cpp ../matrix/naive.cpp -o expNaive
g++ testMatrOps.cpp ../matrix/naive.cpp -o matrOpsNaive
g++ testMinMax.cpp ../matrix/naive.cpp -o minMaxNaive
g++ testBasicOperators.cpp ../matrix/optim.cpp -o basicOptim -std=c++17 -pthread -I/usr/include -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -mavx2 # feel free to edit
g++ testExp.cpp ../matrix/optim.cpp -o expOptim -std=c++17 -pthread -I/usr/include -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -mavx2 # feel free to edit
g++ testMatrOps.cpp ../matrix/optim.cpp -o matrOpsOptim -std=c++17 -pthread -I/usr/include -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -mavx2 # feel free to edit
g++ testMinMax.cpp ../matrix/optim.cpp -o minMaxOptim -std=c++17 -pthread -I/usr/include -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -mavx2 # feel free to edit
