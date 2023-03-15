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
    # Attribute r: the position vector (x, y, z) of the telescope
    # from the origin.
    # Invariant: r is a 3x1 numpy matrix with floats as entries
    #
    # Attribute r_1: the position vector (x, y, z) of the telescope
    # from the Sun.
    # Invariant: r is a 3x1 numpy matrix with floats as entries
    #
    # Attribute r_2: the position vector (x, y, z) of the telescope
    # from the Earth.
    # Invariant: r is a 3x1 numpy matrix with floats as entries
    #
    # Attribute mu: the mass parameter of the dynamical system
    # Invariant: mu is a float
    #
    # Attribute a_1: the acceleration vector (a_x, a_y, a_z) of the
    # telescope due to the Sun
    # Invariant: a_1 is a 3x1 numpy matrix with floats as entries
    #
    # Attribute a_2: the acceleration vector (a_x, a_y, a_z) of the
    # telescope due to the Earth
    # Invariant: a_2 is a 3x1 numpy matrix with floats as entries
    #
    # Attribute psi: the angle made between the vectors to the
    # telescope from the Sun and the Earth
    # Invariant: psi is a float
    #
    # Attribute theta_1: the angle made between the vectors from the
    # telescope to the Sun and the Starshade
    # Invariant: theta_1 is a float
    #
    # Attribute r_p_hat: the unit vector from the telescope to the pole
    # Invariant: r_p_hat is a 3x1 numpy matrix with floats as entries

    def __init__(self, r, mu):
        """
        Args:
            r (3x1 numpy matrix with floats as entries)
                The position vector of the telescope

            mu (float)
                The mass parameter of the dynamical system
        """
        i_hat = np.matrix([0, 0, 1])
        self.r = r
        self.r_1 = (r + (mu * i_hat))
        self.r_2 = (r + ((mu - 1) * i_hat))
        self.mu = mu
        self.a_1 = self.calc_a_1()
        self.a_2 = self.calc_a_2()
        self.psi = self.calc_psi()
        self.theta_1 = self.calc_theta_1()
        self.r_p_hat = self.calc_r_p_hat()

    def calc_a_1(self):
        """ Calculate a_1

        This method calculates the acceleration of the telescope due
        to the gravity of the Sun.

        The equation in this method is based on Equation 2 in the
        referenced paper.

        Returns:
            a_1 (3x1 numpy matrix of floats)
        """
        a_1 = np.divide((self.mu - 1) * self.r_1, np.power(np.linalg.norm(self.r_1), 3))
        return a_1

    def calc_a_2(self):
        """ Calculate a_2

        This method calculates the acceleration of the telescope due
        to the gravity of the Earth.

        The equation in this method is based on Equation 3 in the
        referenced paper.

        Returns:
            a_2 (3x1 numpy matrix of floats)
        """
        a_2 = np.divide(-1 * self.mu * self.r_2, np.power(np.linalg.norm(self.r_2), 3))
        return a_2

    def calc_psi(self):
        """ Calculate psi

        This method calculates the angle made between the vectors to the
        telescope from the Sun and the Earth.

        The equation in this method is based on Equation 17 in the
        referenced paper.

        Returns:
            psi (float)
        """
        psi = np.arccos(np.divide(np.power(self.r_1, 2) + np.power(self.r_2) - 1,
                                  2 * np.matmul(self.r_1, self.r_2)))
        return psi

    def calc_theta_1(self):
        """ Calculate theta_1

        This method calculates the angle made between the vectors to the
        telescope from the Sun vector from the telescope to the Starshade.

        The equation in this method is based on Equation 19 in the
        referenced paper.

        Returns:
            theta_1 (float)
        """
        numerator = np.sin(2 * self.psi) * np.divide(self.a_2, self.r_2)
        denominator = np.divide(self.a_1, self.r_1) + np.cos(2 * self.psi) * \
                      np.divide(self.a_2, self.r_2)

        theta_1 = .5 * np.arctan(numerator, denominator)
        return theta_1

    def calc_r_p_hat(self):
        """ Calculate r_p_hat

        This method calculates the unit vector from the telescope to the pole.

        The equation in this method is the derived from rotation described on
        page 7 of the referenced paper and the wikipedia article on Rotation
        Matrices (https://en.wikipedia.org/wiki/Rotation_matrix)

        Returns:
            r_p_hat ((3x1 numpy matrix of floats)
        """
        j_hat = np.matrix([0, 1, 0])
        k_hat = np.matrix([0, 0, 1])
        u = np.matrix([0, (-1 * self.r[2] * j_hat), (self.r[1] * k_hat)])
        I = np.identity(3)

        rotation_matrix = (np.cos(self.theta_1) * I) + \
                          (np.sin(self.theta_1) * np.cross(u, -1 * I)) + \
                          (1 - np.cos(self.theta_1) * np.outer(u, u))

        scipy_rotation = R.from_matrix(rotation_matrix)
        r_p = scipy_rotation.apply(self.r)
        r_p_hat = np.divide(r_p, np.norm(r_p))
        return r_p_hat

    def get_pole(self):
        """ Get r_p_hat

        Get the unit vector from the telescope to the pole.

        Returns:
            r_p_hat ((3x1 numpy matrix of floats)
        """
        return self.r_p_hat
