from PhageIPSeq_CFS.helpers import get_data_with_outcome
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection


def identify_statistically_different_oligos():
    df = get_data_with_outcome(data_type='exist')
    # preform chi-test to compare groups
    df = df.reset_index(0).groupby('is_CFS').agg(['sum', 'count']).stack(-1).T
    df = df.stack(0)
    df['diff'] = df['count'] - df['sum']
    df = df.unstack(-1)
    df['fisher_exact_p_vals'] = df.apply(
        lambda row: fisher_exact([[row[('sum', 0)], row[('diff', 0)]],
                                  [row[('sum', 1)], row[('diff', 1)]]])[1], axis=1)
    if fdrcorrection(df['fisher_exact_p_vals'])[0].any():
        print(
            f"A total of {fdrcorrection(df['fisher_exact_p_vals'])[0].sum()} oligos passed FDR for fisher's exact test")
    else:
        print(f"No oligos passed FDR for fisher's exact test for {len(df)} oligos")


if __name__ == "__main__":
    identify_statistically_different_oligos()
