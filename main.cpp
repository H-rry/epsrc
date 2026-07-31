#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include <cmath>

// AMGCL Headers
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp> // Or cg.hpp, gmres.hpp

// pybind Headers
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using SparseMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

// Define AMGCL Solver Type using C++ template composition
using Backend = amgcl::backend::builtin<double>;
using AMGCL_Solver = amgcl::make_solver<
    amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0
    >,
    amgcl::solver::bicgstab<Backend>
>;


class SystemBuilder {
public:
    explicit SystemBuilder(int N) : N_(N), total_cols_(5 * N + 3) {}

    SparseMat assemble_A() {
        int total_rows = 5 * N_ + 3;
        double k = 1;
        double lambda = 1; // ratio of viscocity of the fluids. i.e. don't pick one. [0.2, 0.5, 2, 5]

        SparseMat A(total_rows, total_cols_);

        // Reserve up to 6 non-zeros per row - revise later maybe because they have ~ 4 on average
        Eigen::VectorXi nnz_per_row(total_rows);
        nnz_per_row.setConstant(6); 
        A.reserve(nnz_per_row);

        // Block N1
        for (int i = 0; i < N_; ++i) {
            double N_one_coeff = 2.0 * (2.0 * (i + 1) + 1.0) * (1.0/((i + 1) * ((i + 1) + 1.0)));
            add(A, i, 5 * i, -N_one_coeff * (i + 1) * ((i + 1) + 2.0) * lambda);
            add(A, i, 5 * i + 1, -N_one_coeff * ((i + 1) * (i + 1) - 1.0) * lambda);
            add(A, i, 5 * i + 2, -N_one_coeff * ((i + 1) * (i + 1) - 1.0) * lambda);
            add(A, i, 5 * i + 3, -N_one_coeff * (i + 1) * ((i + 1) + 2.0) * lambda);
            if (i != 0) {add(A, i, 5 * i + 4, -N_one_coeff * (((i + 1) * (i + 1) - 1.0) - (i + 1) * ((i + 1) + 2.0)));}
        }
        add(A, 0, 5 * N_, 9.0);


        // Block N2 - cascading sets of quadrouple 1's
        for (int i = 0; i < N_; ++i) {
            int r = N_ + i;
            add(A, r, 5 * i, 1.0);
            add(A, r, 5 * i + 1, 1.0);
            add(A, r, 5 * i + 2, 1.0);
            add(A, r, 5 * i + 3, 1.0);
        }

        // Block N3  
        int r = 2 * N_;
        // n = 1  
        add(A, r, 0, -4.0);
        add(A, r, 1, -2.0);
        add(A, r, 2, -1.0);
        add(A, r, 3, 1.0);
        add(A, r, 5 * N_, -2.0);
        add(A, r, 5 * N_ + 1, -2.0/3.0);
        // n >= 2
        for (int i = 1; i < N_; ++i) {
            add(A, r + i, 5 * i, -((i + 1) + 3.0));
            add(A, r + i, 5 * i + 1, -((i + 1) + 1.0));
            add(A, r + i, 5 * i + 2, -(2-(i + 1)));
            add(A, r + i, 5 * i + 3, (i + 1));
            add(A, r + i, 5 * i + 4, 2.0);
        }

        // Block N4
        r = 3 * N_;
        // n = 1  
            add(A, r, 0, 6.0 * std::pow(k, 2.0));
            add(A, r, 1, 3.0);
            add(A, r, 2, 1.5 * std::pow(k, -1.0));
            add(A, r, 3, -1.5 * std::pow(k, -3.0));
            add(A, r, 5 * N_ + 2, -1.0);
        // n >= 2
        for (int i = 1; i < N_; ++i) {
            add(A, r + i, 5 * i, ((i + 1) + 3.0) * std::pow(k, (i + 1) + 1.0));
            add(A, r + i, 5 * i + 1, ((i + 1) + 1.0) * std::pow(k, (i + 1) - 1.0));
            add(A, r + i, 5 * i + 2, (-(i + 1) + 2.0) * std::pow(k, -(i + 1)));
            add(A, r + i, 5 * i + 3, -(i + 1) * std::pow(k, -(i + 1) - 2.0));            
        }

        // Block N5
        r = 4 * N_;
        // n = 1  
            add(A, r, 0, 3.0 * std::pow(k, 2.0));
            add(A, r, 1, 3.0);
            add(A, r, 2, 3.0 * std::pow(k, -1.0));
            add(A, r, 3, 3.0 * std::pow(k, -3.0));
            add(A, r, 5 * N_ + 2, -1.0);
        // n >= 2
        for (int i = 1; i < N_; ++i) {
            add(A, r + i, 5 * i, std::pow(k, (i + 1) + 1.0));
            add(A, r + i, 5 * i + 1, std::pow(k, (i + 1) - 1.0));
            add(A, r + i, 5 * i + 2, std::pow(k, -(i + 1)));
            add(A, r + i, 5 * i + 3, std::pow(k, -(i + 1) - 2.0));            
        }

        // Block Row 1
        add(A, 5 * N_, 4, 1.0);
        add(A, 5 * N_, 5 * N_ + 1, -1.0/3.0);
        add(A, 5 * N_, 5 * N_, 1.0);
        
        // Block Row 2
        add(A, 5 * N_ + 1, 4, 1.0);

        // Block Row 3
        add(A, 5 * N_ + 2, 2, 1.0);

    
        std::cout << "Successfully assembled " << A.rows() << "x" << A.cols() 
              << " matrix with " << A.nonZeros() << " non-zero entries.\n";

        A.makeCompressed();
        return A;
    }

    Eigen::VectorXd assemble_b() {
    int total_rows = 5 * N_ + 3;
    double Ma = 1.0;

    Eigen::VectorXd b = Eigen::VectorXd::Zero(total_rows);

    for (int i = 1; i < N_; ++i) {
        b(i) = Ma * std::pow(0.1, i + 1);
    }

    b(5 * N_ + 2) = -(1.0 / 12.0) * M_1_PI;

    return b;
}


    // Helper to print full dense grid (for small N <= 3)
    void printDense(const SparseMat& A) const {
        std::cout << "\n--- Dense Matrix View (" << A.rows() << " x " << A.cols() << ") ---\n";
        std::cout << Eigen::MatrixXd(A) << "\n";
    }

    // Helper to print sparse non-zero elements
    void printSparseNonZeros(const SparseMat& A) const {
        std::cout << "\n--- Non-Zero Entries List ---\n";
        for (int r = 0; r < A.rows(); ++r) {
            std::cout << "Row " << r << ": ";
            for (SparseMat::InnerIterator it(A, r); it; ++it) {
                std::cout << "[" << it.col() << "] = " << it.value() << "  ";
            }
            std::cout << "\n";
        }
    }

private:
    inline void add(SparseMat& mat, int r, int c, double val) {
        if (val != 0.0) {
            // .insert() is O(1) when row & column order is strictly respected
            mat.insert(r, c) = val; 
        }
    }

    int N_;
    int total_cols_;
};

Eigen::VectorXd Solve(const SparseMat& A, const Eigen::VectorXd& b){
    boost::property_tree::ptree prm;
    prm.put("solver.tol", 1e-8);     // Target relative residual tolerance
    prm.put("solver.maxiter", 500);  // Max iterations

    AMGCL_Solver solve(amgcl::adapter::eigen(A), prm);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());

    int iters;
    double error;

    std::tie(iters, error) = solve(b, x);

    std::cout << "AMGCL Convergence Status:\n";
    std::cout << "  Iterations: " << iters << "\n";
    std::cout << "  Achieved Error: " << error << "\n";

    return x;
}

Eigen::VectorXd solve_system(int N) {
    SystemBuilder builder(N);
    SparseMat A = builder.assemble_A();
    Eigen::VectorXd b = builder.assemble_b();

    Eigen::VectorXd x = Solve(A, b);

    return x;
}

PYBIND11_MODULE(solver_module, m) {
    m.def("solve_system", &solve_system, "Solves fluid system for given N");
}

int main(int argc, char* argv[]) {
    int N = 2; // Default small test size
    if (argc > 1) N = std::atoi(argv[1]);

    SystemBuilder builder(N);
    SparseMat A = builder.assemble_A();
    Eigen::VectorXd b = builder.assemble_b();

    Solve(A, b);

    return 0;
}