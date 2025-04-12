M = 1000  # Hệ số M lớn dùng trong phương pháp Big-M

def parse_input():
    # Ví dụ: Max Z = x1 - x2 + 2x3
    # Ràng buộc:
    # x1 - x2 + 2x3 = 3
    # 3x1 + 4x2 - 2x3 = 5
    c = [1, -1, 2]
    A = [
        [1, -1, 2],
        [3, 4, -2]
    ]
    b = [3, 5]
    signs = ['=', '=']
    return c, A, b, signs

def build_bigM_tableau(c, A, b, signs):
    num_constraints = len(A)
    num_variables = len(c)

    artificial_vars = []
    rows = []
    basis = []
    basis_cost = []

    for i in range(num_constraints):
        row = A[i][:]
        sign = signs[i]

        # Thêm artificial variable
        artificial_col = [0] * len(artificial_vars) + [1]
        artificial_vars.append(f"x{num_variables + len(artificial_vars) + 1}")
        row += artificial_col

        basis.append(num_variables + len(artificial_vars))  # lưu chỉ số biến cơ sở
        basis_cost.append(-M)  # hệ số là -M (vì đang tìm Max)

        rows.append(row)

    total_vars = num_variables + len(artificial_vars)

    for i in range(num_constraints):
        while len(rows[i]) < total_vars:
            rows[i].append(0)

    # Tính hàng Z: Zj = Σ (cB * aij) - cj
    z_row = [0] * total_vars
    rhs_z = 0
    for i in range(num_constraints):
        coeff = basis_cost[i]
        rhs_z += coeff * b[i]
        for j in range(total_vars):
            z_row[j] += coeff * rows[i][j]

    for j in range(num_variables):
        z_row[j] -= c[j]

    return basis, basis_cost, b, rows, z_row, rhs_z, total_vars

def print_bigM_tableau(basis, basis_cost, b, rows, z_row, rhs_z, var_names):
    print("\nBảng đơn hình ban đầu (phương pháp Big-M):\n")

    # In tiêu đề
    header = ["Biến cơ sở", "Hệ số RHS"] + var_names
    print(" | ".join(f"{h:^12}" for h in header))
    print("-" * (14 * len(header)))

    # In từng dòng ràng buộc
    for i in range(len(basis)):
        var = var_names[basis[i] - 1]
        cost = basis_cost[i]
        label = f"{var} = {cost}"
        row_vals = rows[i]
        row_str = [f"{label:^12}", f"{b[i]:^12}"] + [f"{v:^12}" for v in row_vals]
        print(" | ".join(row_str))

    # In dòng Z
    z_label = "Z"
    z_row_str = [f"{z_label:^12}", f"{rhs_z:^12}"] + [f"{v:^12}" for v in z_row]
    print("-" * (14 * len(header)))
    print(" | ".join(z_row_str))

# ========== CHẠY THỬ ==========
c, A, b, signs = parse_input()
basis, basis_cost, b, all_rows, z_row, rhs_z, total_vars = build_bigM_tableau(c, A, b, signs)
var_names = [f"x{i+1}" for i in range(total_vars)]
print_bigM_tableau(basis, basis_cost, b, all_rows, z_row, rhs_z, var_names)
