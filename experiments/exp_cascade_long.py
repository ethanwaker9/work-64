from common import save_json
from flm.theory import cascade_thresholds

if __name__ == "__main__":
    Ks = cascade_thresholds(0.5, 10)
    print(Ks, flush=True)
    save_json("cascade_long.json", {"0.5": Ks})
