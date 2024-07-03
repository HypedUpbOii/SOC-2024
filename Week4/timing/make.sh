g++ timeBasicOperators.cpp ../matrix/naive.cpp -o basicNaive
g++ timeExp.cpp ../matrix/naive.cpp -o expNaive
g++ timeMatrOps.cpp ../matrix/naive.cpp -o matrOpsNaive
g++ timeMinMax.cpp ../matrix/naive.cpp -o minMaxNaive
g++ timeBasicOperators.cpp ../matrix/optim.cpp -o basicOptim -std=c++17 -pthread -I/usr/include -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -mavx2 # feel free to edit
g++ timeExp.cpp ../matrix/optim.cpp -o expOptim -std=c++17 -pthread -I/usr/include -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -mavx2 # feel free to edit
g++ timeMatrOps.cpp ../matrix/optim.cpp -o matrOpsOptim -std=c++17 -pthread -I/usr/include -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -mavx2 # feel free to edit
g++ timeMinMax.cpp ../matrix/optim.cpp -o minMaxOptim -std=c++17 -pthread -I/usr/include -L/usr/lib/x86_64-linux-gnu/ -lOpenCL -mavx2 # feel free to edit
