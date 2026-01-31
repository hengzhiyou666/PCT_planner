#pragma once

#include <osqp/osqp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// OSQP 类型别名：OSQP 使用 c_float 和 c_int，这里定义为 OSQPFloat 和 OSQPInt 以保持代码兼容性
#ifndef OSQPFloat
#define OSQPFloat c_float
#endif
#ifndef OSQPInt
#define OSQPInt c_int
#endif
#ifndef OSQPCscMatrix
#define OSQPCscMatrix csc
#endif
// OSQPSolver 是 OSQPWorkspace 的别名（旧版本兼容）
#ifndef OSQPSolver
typedef OSQPWorkspace OSQPSolver;
#endif

namespace common {

using ColSparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

/**
 * @class OsqpInterface (ADMM based qp solver)
 * @brief solve convex quadratic programming problem
 *
 * min_{x} 1/2 x'Px + q'x
 * s.t.    l <= Ax <= u
 *
 * Note: osqp use sparse matrix representation
 * [compressed-column](https://people.sc.fsu.edu/~jburkardt/data/cc/cc.html)
 */
class OsqpInterface {
 public:
  // please notice that q, l, u may be modified during optimization
  static bool Solve(const ColSparseMatrix& P,
                    Eigen::Ref<Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1>> q,
                    const ColSparseMatrix& A,
                    Eigen::Ref<Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1>> l,
                    Eigen::Ref<Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1>> u,
                    Eigen::VectorXd* x);

  // Although dense matrix interface is provided, one should notice that dense
  // matrix P, A will be convert to sparse matrix anyway.
  static bool Solve(const Eigen::MatrixXd& P,
                    Eigen::Ref<Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1>> q,
                    const Eigen::MatrixXd& A,
                    Eigen::Ref<Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1>> l,
                    Eigen::Ref<Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1>> u,
                    Eigen::VectorXd* x);
};

}  // namespace common