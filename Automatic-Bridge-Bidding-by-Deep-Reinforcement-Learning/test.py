import scipy
import scipy.io as sio

if __name__ == "__main__":
    mat_content = sio.loadmat("model_valcost_1.063088e-01_totalbid4_4_128_50_3.204176e-03_9.800000e-01_8.200000e-01_alpha1.000000e-01.mat")
    print(mat_content["BB_qlearning"].shape)
    print(mat_content["WW_qlearning"].shape)
    import pdb
    pdb.set_trace()
