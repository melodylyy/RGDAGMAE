import argparse

def get_args():


    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=179, help='train epochs')#
    parser.add_argument('--n_heads', default=2, type=int)
    parser.add_argument('--enc_layers', default=8, type=int)
    parser.add_argument('--GGCN_A', default='mean', type=str)
    parser.add_argument('--aggregator_type', default='gcn', type=str)
    parser.add_argument('--mask_ratio', default=0.3, type=int)
    parser.add_argument('--rito', default=0.3, type=int)





    parser.add_argument('--model', type=str, default='GGMAEA', help='model name')


    parser.add_argument('--neg_ratio', default=1, type=int, choices=[1, 2, 3])
    parser.add_argument('--m_d', default=431, type=int)
    parser.add_argument('--d_d', default=140, type=int)

    parser.add_argument('--miRNA_sim_dir', default=r'.../dataset/m-mmatrix.txt')
    parser.add_argument('--drug_sim_dir', default=r'.../dataset/d-d2matrix.txt')
    parser.add_argument('--association_m_dir', default=r'.../dataset/guanlianmatrix.txt')

    parser.add_argument('--G_weight', default=0.1, type=int)


    parser.add_argument('--d_h', default=256, type=int)


    parser.add_argument('--depth', default=4, type=int)
 



    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--fold', default=5, type=int)


















    args = parser.parse_args()

    return args




