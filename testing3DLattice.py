from Standard_Imports import *

testlattice = Test3D(100, 100)


total_iters = 10000
logging_iters = 10
start_time = time.time()
for i in range(total_iters):
    testlattice.make_update()
    if i % logging_iters == 0 and i != 0:
        end_time = time.time()
        print(f"Completed {logging_iters} iterations in {(end_time-start_time):.3g}s")
        start_time = end_time

testlattice.draw_fields(0)
testlattice.draw_fields(1)
testlattice.draw_fields(2)
