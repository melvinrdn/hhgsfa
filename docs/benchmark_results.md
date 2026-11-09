Julia compile time 3.3 s (excluded below).


### Accuracy

| window | points per cycle | hhgsfa, own convergence | julia, own convergence | between codes | between codes, phase [rad] |
|---|---|---|---|---|---|
| first return | 128 | 5.0e-05 | 2.7e-02 | 1.2e-01 | 3.6e-02 |
| first return | 256 | 8.2e-06 | 1.1e-02 | 1.3e-01 | 4.0e-02 |
| first return | 512 | 7.5e-07 | 4.0e-03 | 1.4e-01 | 4.2e-02 |
| first return | 1024 | reference | reference | 1.4e-01 | 4.3e-02 |
| every return | 128 | 5.3e-04 | 5.3e-04 | 1.9e-05 | 2.7e-05 |
| every return | 256 | 1.6e-04 | 1.6e-04 | 1.2e-06 | 1.7e-06 |
| every return | 512 | reference | reference | 7.4e-08 | 1.1e-07 |


### Run time

Measured on 13th Gen Intel(R) Core(TM) i7-13700, 24 logical cores.

| case | time points | julia, 1 thread [s] | hhgsfa numpy [s] | hhgsfa numba, 1 thread [s] | hhgsfa numba, 24 threads [s] |
|---|---|---|---|---|---|
| 10 fs, first return, 128 pts/cycle | 1491 | 0.21 | 0.012 (×18) | 0.003 (×82) | 0.001 (×292) |
| 10 fs, first return, 256 pts/cycle | 2981 | 0.88 | 0.052 (×17) | 0.012 (×75) | 0.003 (×316) |
| 10 fs, first return, 512 pts/cycle | 5961 | 3.66 | 0.189 (×19) | 0.045 (×80) | 0.011 (×330) |
| 10 fs, first return, 1024 pts/cycle | 11922 | 15.03 | 0.641 (×23) | 0.160 (×94) | 0.038 (×393) |
| 10 fs, every return, 128 pts/cycle | 1491 | 1.29 | 0.122 (×11) | 0.019 (×67) | 0.005 (×244) |
| 10 fs, every return, 256 pts/cycle | 2981 | 5.41 | 0.445 (×12) | 0.072 (×75) | 0.024 (×224) |
| 10 fs, every return, 512 pts/cycle | 5961 | 22.24 | 1.742 (×13) | 0.221 (×101) | 0.086 (×258) |


### Harmonic contrast, 30 fs at 1.5e14 W/cm²

Amplitude of each harmonic over the larger of its two even neighbors.

| window | code | H17 | H19 | H21 | H31 | H41 |
|---|---|---|---|---|---|---|
| first return | hhgsfa | 0.6 | 0.9 | 1.0 | 1.2 | 1.1 |
| first return | julia | 0.7 | 1.0 | 1.0 | 1.2 | 1.1 |
| every return | hhgsfa | 0.7 | 0.6 | 0.3 | 1.0 | 1.1 |
| every return | julia | 0.7 | 0.6 | 0.3 | 1.0 | 1.1 |

