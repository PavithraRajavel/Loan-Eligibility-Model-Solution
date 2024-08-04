import seaborn as sns
import matplotlib.pyplot as plt


def plot_loan_approval_distribution(df):
    sns.countplot(x='Loan_Status', data=df)
    plt.title('Loan Status Distribution')
    plt.show()


def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing > 0
    missing.sort_values(inplace=True)
    missing.plot.bar()
    plt.title('Missing Values by Feature')
    plt.show()
