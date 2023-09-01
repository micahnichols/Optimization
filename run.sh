ulimit -s unlimited
export OMP_STACKSIZE=10485760
./nn_calib_gcc_new -in temp.input > temp.output
