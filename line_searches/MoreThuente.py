import numpy as np
from line_searches.utils import create_phi, create_omega, find_kink_step_indices, project_gradient


class LineSearchException(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code
        super().__init__(self.message)


class MoreThuente:
    def __init__(self, f_tol=1e-4, gtol=0.9, x_tol=1e-8, alphamin=1e-16, alphamax=65536.0, maxfev=100, alpha0=1.0):
        self.f_tol = f_tol  # c_1 Wolfe sufficient decrease condition
        self.gtol = gtol  # c_2 Wolfe curvature condition (Recommend 0.1 for GradientDescent)
        self.x_tol = x_tol
        self.alphamin = alphamin
        self.alphamax = alphamax
        self.maxfev = maxfev
        self.alpha0 = alpha0

    # fn, fn_prime, x, gradient_at_x
    def search(self, fn, fn_prime, x, gradient_at_x):
        f_tol, gtol, x_tol, alphamin, alphamax, maxfev = (
            self.f_tol, self.gtol, self.x_tol, self.alphamin, self.alphamax, self.maxfev)

        constraints = (-1e12, 1e12)
        phi, dphi_minus, dphi_plus = create_phi(fn, fn_prime, x, gradient_at_x, constraints)

        phi_0 = phi(0)
        dphi_0 = dphi_plus(0)

        iterfinitemax = -np.log2(np.finfo(float).eps)
        info = 0
        info_cstep = 1  # Info from step

        zeroT = 0.0
        alpha = self.alpha0

        # Check the input parameters for errors.
        if alpha <= zeroT or f_tol < zeroT or gtol < zeroT or x_tol < zeroT or alphamin < zeroT or alphamax < alphamin or maxfev <= zeroT:
            raise LineSearchException("Invalid parameters to MoreThuente.", 0)

        if dphi_0 >= zeroT:
            raise LineSearchException("Search direction is not a direction of descent.", 0)

        # Initialize local variables.
        bracketed = False
        stage1 = True
        nfev = 0
        finit = phi_0
        dgtest = f_tol * dphi_0
        width = alphamax - alphamin
        width1 = 2 * width

        # Keep this across calls
        stx = zeroT
        fx = finit
        dgx = dphi_0
        sty = zeroT
        fy = finit
        dgy = dphi_0

        # START: Ensure that the initial step provides finite function values
        if not np.isfinite(alpha):
            alpha = 1.0
        stmin = stx
        stmax = alpha + 4 * (alpha - stx)  # Why 4?
        alpha = max(alpha, alphamin)
        alpha = min(alpha, alphamax)

        f, dg = phi(alpha), dphi_plus(alpha)
        nfev += 1  # This includes calls to f() and g!()
        iterfinite = 0
        while not (np.isfinite(f) and np.isfinite(dg)) and iterfinite < iterfinitemax:
            iterfinite += 1
            alpha /= 2
            f, dg = phi(alpha), dphi_plus(alpha)
            nfev += 1  # This includes calls to f() and g!()
            # Make stmax = (3/2)*alpha < 2alpha in the first iteration below
            stx = (7 / 8) * alpha

        # END: Ensure that the initial step provides finite function values

        while True:
            # Set the minimum and maximum steps to correspond
            # to the present interval of uncertainty.
            if bracketed:
                stmin = min(stx, sty)
                stmax = max(stx, sty)
            else:
                stmin = stx
                stmax = alpha + 4 * (alpha - stx)  # Why 4?

            # Ensure stmin and stmax (used in cstep) don't violate alphamin and alphamax
            stmin = max(alphamin, stmin)
            stmax = min(alphamax, stmax)

            # Force the step to be within the bounds alphamax and alphamin
            alpha = max(alpha, alphamin)
            alpha = min(alpha, alphamax)

            # If an unusual termination is to occur then let alpha be the lowest point obtained so far.
            if (bracketed and (alpha <= stmin or alpha >= stmax)) or nfev >= maxfev - 1 or info_cstep == 0 or (
                    bracketed and stmax - stmin <= x_tol * stmax):
                alpha = stx

            # Evaluate the function and gradient at alpha and compute the directional derivative.
            f, dg = phi(alpha), dphi_plus(alpha)
            nfev += 1  # This includes calls to f() and g!()

            if np.isclose(dg, 0, atol=np.finfo(float).eps):  # Should add atol value to MoreThuente
                return alpha, f

            ftest1 = finit + alpha * dgtest

            # Test for convergence.
            if (bracketed and (alpha <= stmin or alpha >= stmax)) or info_cstep == 0:
                info = 6
            if alpha == alphamax and f <= ftest1 and dg <= dgtest:
                info = 5
            if alpha == alphamin and (f > ftest1 or dg >= dgtest):
                info = 4
            if nfev >= maxfev:
                info = 3
            if bracketed and stmax - stmin <= x_tol * stmax:
                info = 2
            if f <= ftest1 and abs(dg) <= -gtol * dphi_0:
                info = 1

            # Check for termination.
            if info != 0:
                return alpha, f

            # In the first stage we seek a step for which the modified function has a nonpositive value and nonnegative derivative.
            if stage1 and f <= ftest1 and dg >= min(f_tol, gtol) * dphi_0:
                stage1 = False

            # A modified function is used to predict the step only if we have not obtained a step for which the modified function has a nonpositive function value and nonnegative derivative, and if a lower function value has been obtained but the decrease is not sufficient.
            if stage1 and f <= fx and f > ftest1:
                # Define the modified function and derivative values.
                fm = f - alpha * dgtest
                fxm = fx - stx * dgtest
                fym = fy - sty * dgtest
                dgm = dg - dgtest
                dgxm = dgx - dgtest
                dgym = dgy - dgtest
                # Call cstep to update the interval of uncertainty and to compute the new step.
                stx, fxm, dgxm, sty, fym, dgym, alpha, fm, dgm, bracketed, info_cstep = self.cstep(stx, fxm, dgxm, sty,
                                                                                                   fym, dgym, alpha, fm,
                                                                                                   dgm, bracketed,
                                                                                                   stmin, stmax)
                # Reset the function and gradient values for f.
                fx = fxm + stx * dgtest
                fy = fym + sty * dgtest
                dgx = dgxm + dgtest
                dgy = dgym + dgtest
            else:
                # Call cstep to update the interval of uncertainty and to compute the new step.
                stx, fx, dgx, sty, fy, dgy, alpha, f, dg, bracketed, info_cstep = self.cstep(stx, fx, dgx, sty, fy, dgy,
                                                                                             alpha, f, dg, bracketed,
                                                                                             stmin, stmax)

            # Force a sufficient decrease in the size of the interval of uncertainty.
            if bracketed:
                if abs(sty - stx) >= (2 / 3) * width1:
                    alpha = stx + (sty - stx) / 2
                width1 = width
                width = abs(sty - stx)

    def cstep(self, stx, fx, dgx, sty, fy, dgy, alpha, f, dg, bracketed, alphamin, alphamax):
        zeroT = 0.0
        info = 0

        # Check the input parameters for error
        if (bracketed and (alpha <= min(stx, sty) or alpha >= max(stx, sty))) or dgx * (
                alpha - stx) >= zeroT or alphamax < alphamin:
            raise ValueError("Minimizer not bracketed")

        # Determine if the derivatives have opposite sign
        sgnd = dg * (dgx / abs(dgx))

        # First case. A higher function value. The minimum is bracketed. If the cubic step is closer to stx than the quadratic step, the cubic step is taken, else the average of the cubic and quadratic steps is taken.
        if f > fx:
            info = 1
            bound = True
            theta = 3 * (fx - f) / (alpha - stx) + dgx + dg
            s = max(abs(theta), abs(dgx), abs(dg))
            gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (dgx / s) * (dg / s)))
            if alpha < stx:
                gamma = -gamma
            p = (gamma - dgx) + theta
            q = ((gamma - dgx) + gamma) + dg
            r = p / q
            stpc = stx + r * (alpha - stx)
            stpq = stx + ((dgx / ((fx - f) / (alpha - stx) + dgx)) / 2) * (alpha - stx)
            if abs(stpc - stx) < abs(stpq - stx):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2
            bracketed = True

        # Second case. A lower function value and derivatives of opposite sign. The minimum is bracketed. If the cubic step is closer to stx than the quadratic (secant) step, the cubic step is taken, else the quadratic step is taken.
        elif sgnd < 0.0:
            info = 2
            bound = False
            theta = 3 * (fx - f) / (alpha - stx) + dgx + dg
            s = max(abs(theta), abs(dgx), abs(dg))
            gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (dgx / s) * (dg / s)))
            if alpha > stx:
                gamma = -gamma
            p = (gamma - dg) + theta
            q = ((gamma - dg) + gamma) + dgx
            r = p / q
            stpc = alpha + r * (stx - alpha)
            stpq = alpha + (dg / (dg - dgx)) * (stx - alpha)
            if abs(stpc - alpha) > abs(stpq - alpha):
                stpf = stpc
            else:
                stpf = stpq
            bracketed = True

        # Third case. A lower function value, derivatives of the same sign, and the magnitude of the derivative decreases.
        elif abs(dg) < abs(dgx):
            info = 3
            bound = True
            theta = 3 * (fx - f) / (alpha - stx) + dgx + dg
            s = max(abs(theta), abs(dgx), abs(dg))
            gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (dgx / s) * (dg / s)))
            if alpha > stx:
                gamma = -gamma
            p = (gamma - dg) + theta
            q = (gamma + (dgx - dg)) + gamma
            r = p / q
            if r < 0.0 and gamma != 0.0:
                stpc = alpha + r * (stx - alpha)
            elif alpha > stx:
                stpc = alphamax
            else:
                stpc = alphamin
            stpq = alpha + (dg / (dg - dgx)) * (stx - alpha)
            if bracketed:
                if abs(alpha - stpc) < abs(alpha - stpq):
                    stpf = stpc
                else:
                    stpf = stpq
            else:
                if abs(alpha - stpc) > abs(alpha - stpq):
                    stpf = stpc
                else:
                    stpf = stpq

        # Fourth case. A lower function value, derivatives of the same sign, and the magnitude of the derivative does not decrease. If the minimum is not bracketed, the step is either alpha_min or alpha_max, else the cubic step is taken.
        else:
            info = 4
            bound = False
            if bracketed:
                theta = 3 * (f - fy) / (sty - alpha) + dgy + dg
                s = max(abs(theta), abs(dgy), abs(dg))
                gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (dgy / s) * (dg / s)))
                if alpha > sty:
                    gamma = -gamma
                p = (gamma - dg) + theta
                q = ((gamma - dg) + gamma) + dgy
                r = p / q
                stpc = alpha + r * (sty - alpha)
                stpf = stpc
            elif alpha > stx:
                stpf = alphamax
            else:
                stpf = alphamin

        # Update the interval of uncertainty. This update does not depend on the new step or the case analysis above.
        if f > fx:
            sty = alpha
            fy = f
            dgy = dg
        else:
            if sgnd < 0.0:
                sty = stx
                fy = fx
                dgy = dgx
            stx = alpha
            fx = f
            dgx = dg

        # Compute the new step and safeguard it.
        stpf = min(alphamax, stpf)
        stpf = max(alphamin, stpf)
        alpha = stpf
        if bracketed and bound:
            if sty > stx:
                alpha = min(stx + 0.66 * (sty - stx), alpha)
            else:
                alpha = max(stx + 0.66 * (sty - stx), alpha)

        return stx, fx, dgx, sty, fy, dgy, alpha, f, dg, bracketed, info




