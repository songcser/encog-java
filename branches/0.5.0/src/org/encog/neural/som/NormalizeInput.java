/*
  * Encog Neural Network and Bot Library for Java v0.5
  * http://www.heatonresearch.com/encog/
  * http://code.google.com/p/encog-java/
  * 
  * Copyright 2008, Heaton Research Inc., and individual contributors.
  * See the copyright.txt in the distribution for a full listing of 
  * individual contributors.
  *
  * This is free software; you can redistribute it and/or modify it
  * under the terms of the GNU Lesser General Public License as
  * published by the Free Software Foundation; either version 2.1 of
  * the License, or (at your option) any later version.
  *
  * This software is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  * Lesser General Public License for more details.
  *
  * You should have received a copy of the GNU Lesser General Public
  * License along with this software; if not, write to the Free
  * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
  */
package org.encog.neural.som;

import org.encog.matrix.Matrix;
import org.encog.matrix.MatrixMath;


/**
 * NormalizeInput: Input into a Self Organizing Map must be normalized.
 */
public class NormalizeInput {

	/**
	 * This class support two normalization types. Z-AXIS is the most commonly
	 * used normalization type. Multiplicative is used over z-axis when the
	 * values are in very close range.
	 * 
	 * @author jheaton
	 * 
	 */
	public enum NormalizationType {
		Z_AXIS, MULTIPLICATIVE
	}

	/**
	 * What type of normalization should be used.
	 */
	private final NormalizationType type;

	/**
	 * The normalization factor.
	 */
	protected double normfac;

	/**
	 * The synthetic input.
	 */
	protected double synth;

	/**
	 * The input expressed as a matrix.
	 */
	protected Matrix inputMatrix;

	/**
	 * Normalize an input array into a matrix. The resulting matrix will have
	 * one extra column that will be occupied by the synthetic input.
	 * 
	 * @param input
	 *            The input array to be normalized.
	 * @param type
	 *            What type of normalization to use.
	 */
	public NormalizeInput(final double input[], final NormalizationType type) {
		this.type = type;
		calculateFactors(input);
		this.inputMatrix = this.createInputMatrix(input, this.synth);
	}

	/**
	 * Create an input matrix that has enough space to hold the extra synthetic
	 * input.
	 * 
	 * @param pattern
	 *            The input pattern to create.
	 * @param extra
	 *            The synthetic input.
	 * @return A matrix that contains the input pattern and the synthetic input.
	 */
	protected Matrix createInputMatrix(final double pattern[],
			final double extra) {
		final Matrix result = new Matrix(1, pattern.length + 1);
		for (int i = 0; i < pattern.length; i++) {
			result.set(0, i, pattern[i]);
		}

		result.set(0, pattern.length, extra);

		return result;
	}

	/**
	 * Get the resulting input matrix.
	 * @return The resulting input matrix.
	 */
	public Matrix getInputMatrix() {
		return this.inputMatrix;
	}

	/**
	 * The normalization factor.
	 * @return The normalization factor.
	 */
	public double getNormfac() {
		return this.normfac;
	}

	/**
	 * The synthetic input.
	 * @return The synthetic input.
	 */
	public double getSynth() {
		return this.synth;
	}

	/**
	 * Determine both the normalization factor and the synthetic input for the
	 * given input.
	 * 
	 * @param input
	 *            The input to normalize.
	 */
	protected void calculateFactors(final double input[]) {

		final Matrix inputMatrix = Matrix.createColumnMatrix(input);
		double len = MatrixMath.vectorLength(inputMatrix);
		len = Math.max(len, SelfOrganizingMap.VERYSMALL);
		final int numInputs = input.length;

		if (this.type == NormalizationType.MULTIPLICATIVE) {
			this.normfac = 1.0 / len;
			this.synth = 0.0;
		} else {
			this.normfac = 1.0 / Math.sqrt(numInputs);
			final double d = numInputs - Math.pow(len,2);
			if (d > 0.0) {
				this.synth = Math.sqrt(d) * this.normfac;
			} else {
				this.synth = 0;
			}
		}
	}
}