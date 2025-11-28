import argparse
import sys
import time

sys.path.append("..")
import rtde.rtde as rtde


def test_frequency(recipe, freq):
    def __send_output_recipe_and_start_sync(con, recipe, frequency=125):
        if not con.send_output_setup(recipe, frequency=frequency):
            raise Exception("Failed to configure RTDE output")
        if not con.send_start():
            raise Exception("Failed to start RTDE data synchronization")

    con = rtde.RTDE("10.54.253.237", 30004)
    con.connect()
    __send_output_recipe_and_start_sync(con, recipe, freq)
    samples = 0
    rcv_seconds = 2
    expected_samples = freq * rcv_seconds
    start = time.time()
    while samples < expected_samples:
        con.receive()
        samples += 1
    dt = time.time() - start
    freq_actual = samples / dt
    print(freq_actual)
    con.send_pause()


if __name__ == "__main__":
    # create the parser for script
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="output_int_register_0")
    parser.add_argument("--freq", default=125)
    args = vars(parser.parse_args())
    recipe = [args.get("recipe")]
    freq = float(args.get("freq"))

    test_frequency(recipe, freq)
