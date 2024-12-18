import argparse
import os
from collections import OrderedDict
from tqdm import tqdm
import lightning as lit
from utils.connectome_reader import ConnectomeReader, REWIRE_NO_LOOPS_NO_MULTIPLE, REWIRE_NO_LOOPS_MULTIPLE, \
    REWIRE_LOOPS_NO_MULTIPLE, REWIRE_LOOPS_MULTIPLE


def main(args):
    lit.seed_everything(seed=args.seed)
    if isinstance(args.rewiring_p, float):
        args.rewiring_p = [args.rewiring_p]

    pbar = tqdm(range(0, len(args.rewiring_p) * args.n_graphs),
                postfix=OrderedDict({"p": args.rewiring_p[0], "seed": args.seed, "graph_n": 0}))
    for i, p in enumerate(args.rewiring_p):
        for j in range(0, args.n_graphs):
            cr = ConnectomeReader(args.graph_path)
            seed = args.seed + args.n_graphs * i + j
            cr.read(sym_flag=args.rewiring_option, pp=p, seed=seed)

            out_dir = os.path.join(args.out_dir, f"{p * 100}")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"celegans_rewired_n{j}.graphml"), "w") as fp:
                cr.graph_el.save(fp, format='graphml')
            pbar.set_postfix(OrderedDict({"p": p, "seed": seed, "graph_n": j}), refresh=True)
            pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, default="data/connectomes/celegans.graphml",
                        help="Path to the graph file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="data/connectomes/rewired/",
                        help="Path to the output directory")
    parser.add_argument("--n_graphs", type=int, default=10, help="Number of rewired graphs to generate")
    parser.add_argument("--rewiring_p", type=float, default=0.5, nargs='+', help="Rewiring probability")
    parser.add_argument("--rewiring_option", type=int, default=REWIRE_NO_LOOPS_NO_MULTIPLE,
                        help="Rewiring options", choices=[REWIRE_NO_LOOPS_NO_MULTIPLE, REWIRE_NO_LOOPS_MULTIPLE,
                                                          REWIRE_LOOPS_NO_MULTIPLE, REWIRE_LOOPS_MULTIPLE])
    main(parser.parse_args())
