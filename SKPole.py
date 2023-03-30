from scipy.spatial.transform import Rotation as R
import numpy as np


class SKPole:
    """ Stationkeeping Pole Calculator

    A tool for numerically calculates points where the
    differential lateral acceleration of a starshade from the
    reference point of a telescope is approximately zero.

    In the coordinate system of this class, the Sun is located
    at (-1 * mu) on the x-axis (-i_hat) and the Earth is located at
    (1 - mu) on the x-axis. Note that "mu" is the mass parameter of
    this system
    """

    # ATTRIBUTES
    # Attribute i_hat, j_hat, k_hat: unit vectors in R^3
    # Invariant: each unit vector is a 3x1 numpy array
    #
    # Attribute r_t: the position vector (x, y, z) of the telescope
    # from the origin.
    # Invariant: r_t is a 3x1 numpy array with floats as entries
    #
    # Attribute r_1: the position vector (x, y, z) to the telescope
    # from the Sun.
    # Invariant: r_1 is a 3x1 numpy array with floats as entries
    #
    # Attribute r_2: the position vector (x, y, z) to the telescope
    # from the Earth.
    # Invariant: r_2 is a 3x1 numpy array with floats as entries
    #
    # Attribute mu: the mass parameter of the dynamical system
    # Invariant: mu is a float

    def __init__(self, r_t, mu):
        """
        Args:
            r_t (3x1 numpy array with floats as entries)
                The position vector of the telescope

            mu (float)
                The mass parameter of the dynamical system
        """
        self.i_hat = np.array([1, 0, 0])
        self.j_hat = np.array([0, 1, 0])
        self.k_hat = np.array([0, 0, 1])
        self.r_t = r_t
        self.r_1 = (r_t + (mu * self.i_hat))
        self.r_2 = (r_t + ((mu - 1) * self.i_hat))
        self.mu = mu

# =============================================================================
# Internal Mathematical Methods
# =============================================================================

    def calc_a_1(self, r):
        """ Calculate a_1

        This method calculates the acceleration of a satellite due
        to the gravity of the Sun.

        The equation in this method is based on Equation (2) in the
        referenced paper.

        Args:
            r (3x1 numpy array of floats)
                Position vector of satellite

        Returns:
            a_1 (3x1 numpy array of floats)
        """
        a_1 = np.divide((self.mu - 1) * (r + (self.mu * self.i_hat)),
                        np.power(np.linalg.norm(r + (self.mu * self.i_hat)), 3))
        return a_1

    def calc_a_2(self, r):
        """ Calculate a_2

        This method calculates the acceleration of a satellite due
        to the gravity of the Earth.

        The equation in this method is based on Equation (3) in the
        referenced paper.

        Args:
            r (3x1 numpy array of floats)
                Position vector of satellite

        Returns:
            a_2 (3x1 numpy array of floats)
        """
        a_2 = np.divide(-1 * self.mu * (r + ((self.mu - 1) * self.i_hat)),
                        np.power(np.linalg.norm(r + ((self.mu - 1) * self.i_hat)), 3))
        return a_2

    def calc_psi(self):
        """ Calculate psi

        This method calculates the angle made between the vectors to the
        telescope from the Sun and the Earth.

        The equation in this method is based on Equation (17) in the
        referenced paper.

        Returns:
            psi (float)
                (In radians)
        """
        r_1_norm = np.linalg.norm(self.r_1)
        r_2_norm = np.linalg.norm(self.r_2)
        psi = np.arccos((np.power(r_1_norm, 2) + np.power(r_2_norm, 2) - 1) /
                        (2 * r_1_norm * r_2_norm))
        return psi

    def calc_theta_1(self):
        """ Calculate theta_1

        This method calculates the angle made between the vectors to the
        telescope from the Sun vector from the telescope to the Starshade.

        The equation in this method is based on Equation (19) in the
        referenced paper.

        Returns:
            theta_1 (float)
                (In radians)
        """
        r_1_norm = np.linalg.norm(self.r_1)
        r_2_norm = np.linalg.norm(self.r_2)
        a_1_norm = np.linalg.norm(self.calc_a_1(self.r_t))
        a_2_norm = np.linalg.norm(self.calc_a_2(self.r_t))
        psi = self.calc_psi()

        numerator = np.sin(2 * psi) * (a_2_norm / r_2_norm)
        denominator = (a_1_norm / r_1_norm) + np.cos(2 * psi) * \
                      (a_2_norm / r_2_norm)

        theta_1 = .5 * np.arctan(numerator / denominator)
        return theta_1

    def calc_r_p(self):
        """ Calculate r_p

        This method calculates the unit vector from the telescope to the pole.

        The equation in this method is the derived from the rotation described on
        page 7 of the referenced paper

        Returns:
            r_p (3x1 numpy array of floats)
        """
        rotation_matrix = self.calc_rotation_matrix(self.calc_theta_1(), self.calc_orthogonal_vector())

        r_1_hat = self.r_1 / np.linalg.norm(self.r_1)
        r_p = np.dot(rotation_matrix, r_1_hat)

        return r_p

    def calc_d_a_l(self, alpha, d):
        """ Calculate d_a_l

        This method calculates the lateral differential acceleration magnitude
        of the telescope-starshade system

        The calculations in this method are derived from equations (4)-(6) in the
        referenced paper, in addition to other material from the 3rd page.
        Note that for clarity, R in equation (5) is referenced here as "d".


        Args:
            alpha (float)
                The angle away from the pole of the desired lateral differential
                acceleration (in radians)

            d (float)
                The distance between the telescope and starshade

        Returns:
            d_a_l (float)
        """
        r_p = self.calc_r_p()
        rotation_matrix = self.calc_rotation_matrix(alpha, self.calc_orthogonal_vector())
        r_rel_hat = np.matmul(rotation_matrix, r_p)

        r_rel = d * r_rel_hat
        r_s = self.r_t + r_rel
        a_s_1 = self.calc_a_1(r_s)
        a_s_2 = self.calc_a_2(r_s)

        a_t_1 = self.calc_a_1(self.r_t)
        a_t_2 = self.calc_a_2(self.r_t)

        a_s = a_s_1 + a_s_2
        a_t = a_t_1 + a_t_2

        # Eq. (4)
        d_a = a_s - a_t

        # Eq. (6)
        d_a_l = np.linalg.norm(d_a - (np.dot(d_a, r_rel_hat) * r_rel_hat))

        return d_a_l

    def calc_rotation_matrix(self, theta, u):
        """ Calculate the 3-d rotation matrix

        This method calculates the matrix of a proper rotation by angle theta
        around axis u, where u is a unit vector in three dimensions. The equation
        for this method is derived from the wikipedia article on Rotation
        Matrices (https://en.wikipedia.org/wiki/Rotation_matrix)

        Args:
            theta (float)
                theta is the angle of rotation in radians

            u (3x1 numpy array)
                u is the axis of rotation

        Returns:
            rotation_matrix (3x3 numpy matrix)
        """
        I = np.identity(3)
        if np.linalg.norm(u) != 1:
            u_hat = np.divide(u, np.linalg.norm(u))
        else:
            u_hat = u

        # Note that the cross product of vector u and -1 * I is the cross product
        # array of vector u_hat
        rotation_matrix = np.array((np.cos(theta) * I) + \
                          (np.sin(theta) * np.cross(u_hat, -1 * I)) + \
                          (1 - np.cos(theta)) * np.outer(u_hat, u_hat))

        return rotation_matrix

    def calc_orthogonal_vector(self):
        """ Calculate an orthogonal vector to r_p

        This method calculate a vector orthogonal to the plane containing the
        telescope, Sun, and Earth. Note that this also implies the vector will
        be orthogonal to r_p.

        Returns:
            u (3x1 numpy array)
        """
        # FIX THIS, Creates 0 vector if r_t only has an x value, need more general
        # solution. Maybe cross product of ambiguous vector?
        u = np.array([0, -1 * self.r_t[2], self.r_t[1]])
        return u

# =============================================================================
# User Getter Functions
# =============================================================================

    def get_pole(self):
        """ Get r_p

        Get the unit vector from the telescope to the pole.

        Returns:
            r_p ((3x1 numpy array of floats)
        """
        return self.calc_r_p()

    def get_d_a_l(self, alpha, d):
        """ Get d_a_l

        Get the lateral differential acceleration magnitude of the
        telescope-starshade system.

        Args:
            alpha (float)
                The angle away from the pole of the desired lateral differential
                acceleration (in radians)

            d (float)
                The distance between the telescope and starshade

        Returns:
            d_a_l (float)
        """
        return self.calc_d_a_l(alpha, d)
