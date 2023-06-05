from grakel.datasets import fetch_dataset
import grakel
import nog_kernel

def gram_of_WL_kernel(graphs):
    sp_kernel = grakel.kernels.WeisfeilerLehman(n_iter=1, base_graph_kernel=grakel.kernels.VertexHistogram, normalize=False)
    gram_mat_sp = sp_kernel.fit_transform(graphs)

    return gram_mat_sp

def gram_of_NoG_kernel(graphs):
    nog_kernel = nog_kernel.NoGKernel()
    gram_mat_sp = nog_kernel.fit_transform(graphs)

    return gram_mat_sp

if __name__=="__main__":

    grakel_ds = fetch_dataset("MUTAG", verbose=False)
    graphs = grakel_ds.data

    print(gram_of_WL_kernel(graphs))

    print(gram_of_NoG_kernel(graphs))
    