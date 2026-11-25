For a problem where x=(n, 1), F = (n, n), c = (n, 1), A=(m, n) and b = (m, 1) the instance file uses matrix and vector representation in the COO format as folows:


```txt
n m
A
row column value
.
.
.
b
row value
.
.
.
F
row column value
.
.
.
c
row value
.
.
.
```

Same data can be passed in the following JSON format but 0-indexed indices:
```json
{
    "number_of_variables": n, 
    "number_of_constraints": m, 
    "A": [
        [row_1, col_1, value_1], 
                ...
        [row_n, col_m, value_nm]
        ], 


    "b": [
        [b1, value_1],
             ...
        [b_m, value_m]
        ], 

    "F": [
        [row_1, col_1, value_1], 
            ...
        [row_n, col_n, value_n]
        ],

    "c": [
        [c_1, value_1],
              ...
        [c_m, value_m]
        ]
}
```

> **Note**: Above `bqp` text format format uses 1-based indeces!
> The `JSON` format uses 0-based indices