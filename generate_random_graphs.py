import argparse

from tqdm import tqdm

from utils.random_graph_utils import generate_erdos_renyi_graph, generate_barabasi_albert_graph, \
    generate_watts_strogatz_graph, analyze_graph


def main(args):
    n_p, e_p, num_n, num_e, k = analyze_graph(connectome_path=args.graph_path)

    seed = args.seed
    p = args.p

    for i in tqdm(range(args.n_graphs), desc="Generating graphs..."):
        generate_erdos_renyi_graph(
            n=num_n,
            p=num_e / (num_n * num_n),
            n_p=n_p,
            e_p=e_p,
            directory=f'{args.out_dir}/ER',
            seed=seed + i
        )
        generate_barabasi_albert_graph(
            n=num_n,
            m=k,
            n_p=n_p,
            e_p=e_p,
            directory=f'{args.out_dir}/BA',
            seed=seed + i
        )
        generate_watts_strogatz_graph(
            n=num_n,
            k=k,
            p=p,
            n_p=n_p,
            e_p=e_p,
            directory=f'{args.out_dir}/WS',
            seed=seed + i
        )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph_path", type=str, default="data/connectomes/celegans.graphml",
                        help="Path to the connectome graphml file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="data/connectomes/random_graph_simulator",)
    parser.add_argument("--p", type=float, default=0.5, help="Probability of rewiring for WS model")
    parser.add_argument("--n_graphs", type=int, default=1, help="Number of graphs")

    main(parser.parse_args())
