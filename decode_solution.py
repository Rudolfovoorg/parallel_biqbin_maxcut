import numpy as np
import json
import argparse
import pandas as pd

def decode_solution(input_json, output_json):
    """
    Decodes the binary solution vector and returns the portfolio metrics.
    """

    metrics = {}
    b_solution = np.array(output_json['qubo']['x'])
    tickers = input_json['tickers']
    stocks_prices = np.array(input_json['prices'])
    mu = np.array(input_json['mu'])
    cov = np.array(input_json['cov'])

    if 'index_map' in input_json:
        index_map = input_json['index_map']

        # Determine number of assets
        N = max(i for i, _ in index_map) + 1
        n_vec = np.zeros(N, dtype=int)
        for bit_value, (i, k) in zip(b_solution, index_map):
            n_vec[i] += bit_value * (2 ** k)

        budget = input_json['budget']
        ni_pi = n_vec*stocks_prices
        wi = ni_pi/budget
        spent = float(np.dot(stocks_prices, n_vec))
        
        n_vec = [int(i) for i in n_vec]


        metrics['budget'] = budget
        metrics['budget_spent'] = spent
    else:
        n_vec = [int(i) for i in b_solution]
        wi = b_solution.copy()

    
    port_ret = float(np.dot(wi,mu))
    port_var = float(np.dot(wi, cov @ wi))
    port_std = float(np.sqrt(port_var))

    metrics['stocks'] = dict(zip(tickers,n_vec))
    metrics['port_ret'] = port_ret
    metrics['port_var'] = port_var
    metrics['port_std'] = port_std

    return metrics

def pretty_print(data: dict):
    portfolio_df = pd.DataFrame([{
    'total_budget:': data['budget'],
    'budget_spent': data['budget_spent'],
    'expected_return': data['port_ret'],
    'variance': data['port_var'],
    'risk': data['port_std']
    }])
    # --- Stocks holdings ---
    stocks_df = pd.DataFrame(list(data['stocks'].items()), columns=['stock', 'quantity'])

    # Show results
    print('Portfolio Summary:')
    print(portfolio_df)
    print('\nStocks Holdings:')
    print(stocks_df.sort_values('quantity', ascending=False))
    

def main():
    parser = argparse.ArgumentParser(
        description='Decode Biqbin solution to portfolio solution.'
    )
    parser.add_argument('qubo', help='Path to qubo file.')
    parser.add_argument('-s', '--solution', help='Path to solution file.')
    
    
    args = parser.parse_args()
    input_file = args.qubo
    output_file = args.solution if args.solution else f'{input_file}.output.json'

    with open(input_file, 'r') as f:
        input_json = json.load(f)
    with open(output_file, 'r') as f:
        output_json = json.load(f)

    metrics = decode_solution(input_json, output_json)
    pretty_print(metrics)
    with open(input_file + '.portfolio_solution.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main() 