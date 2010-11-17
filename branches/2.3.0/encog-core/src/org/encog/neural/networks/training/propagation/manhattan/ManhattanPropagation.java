/*
 * Encog(tm) Artificial Intelligence Framework v2.3
 * Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2010 by Heaton Research Inc.
 * 
 * Released under the LGPL.
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
 * 
 * Encog and Heaton Research are Trademarks of Heaton Research, Inc.
 * For information on Heaton Research trademarks, visit:
 * 
 * http://www.heatonresearch.com/copyright.html
 */

package org.encog.neural.networks.training.propagation.manhattan;

import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.LearningRate;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.gradient.CalculateGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * One problem that the backpropagation technique has is that the magnitude of
 * the partial derivative may be calculated too large or too small. The
 * Manhattan update algorithm attempts to solve this by using the partial
 * derivative to only indicate the sign of the update to the weight matrix. The
 * actual amount added or subtracted from the weight matrix is obtained from a
 * simple constant. This constant must be adjusted based on the type of neural
 * network being trained. In general, start with a higher constant and decrease
 * it as needed.
 * 
 * The Manhattan update algorithm can be thought of as a simplified version of
 * the resilient algorithm. The resilient algorithm uses more complex techniques
 * to determine the update value.
 * 
 * @author jheaton
 * 
 */
public class ManhattanPropagation extends Propagation implements LearningRate {

	/**
	 * The default tolerance to determine of a number is close to zero.
	 */
	static final double DEFAULT_ZERO_TOLERANCE = 0.001;

	/**
	 * The zero tolearnce to use.
	 */
	private final double zeroTolerance;

	/**
	 * The learning rate.
	 */
	private double learningRate;
	
	/**
	 * The gradients.
	 */
	private double[] gradients;

	/**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	private final Logger logger = LoggerFactory.getLogger(this.getClass());

	/**
	 * Construct a class to train with Manhattan propagation. Use default zero
	 * tolerance.
	 * 
	 * @param network
	 *            The network that is to be trained.
	 * @param training
	 *            The training data to use.
	 * @param learnRate
	 *            A fixed learning to the weight matrix for each training
	 *            iteration.
	 */
	public ManhattanPropagation(final BasicNetwork network,
			final NeuralDataSet training, final double learnRate) {
		this(network, training, learnRate,
				ManhattanPropagation.DEFAULT_ZERO_TOLERANCE);
	}

	/**
	 * Construct a Manhattan propagation training object.
	 * 
	 * @param network
	 *            The network to train.
	 * @param training
	 *            The training data to use.
	 * @param learnRate
	 *            The learning rate.
	 * @param zeroTolerance
	 *            The zero tolerance.
	 */
	public ManhattanPropagation(final BasicNetwork network,
			final NeuralDataSet training, final double learnRate,
			final double zeroTolerance) {
		super(network, training);
		this.zeroTolerance = zeroTolerance;
		this.learningRate = learnRate;
		this.gradients = new double[network.getStructure().calculateSize()];
	}

	/**
	 * @return The learning rate that was specified in the constructor.
	 */
	public double getLearningRate() {
		return this.learningRate;
	}

	/**
	 * @return The zero tolerance that was specified in the constructor.
	 */
	public double getZeroTolerance() {
		return this.zeroTolerance;
	}

	/**
	 * Perform a training iteration. This is where the actual Manhattan 
	 * specific training takes place.
	 * 
	 * @param prop
	 *            The gradients.
	 * @param weights
	 *            The network weights.
	 */
	@Override
	public void performIteration(final CalculateGradient prop,
			final double[] weights) {

		this.gradients = prop.getGradients();

		for (int i = 0; i < this.gradients.length; i++) {
			weights[i] += updateWeight(i);
		}

	}

	/**
	 * Set the learning rate.
	 * 
	 * @param rate
	 *            The new learning rate.
	 */
	public void setLearningRate(final double rate) {
		this.learningRate = rate;
	}

	/**
	 * Determine the amount to update a weight by.
	 * @param index The index of the weight to update.
	 * @return The amount the weight should be updated by.
	 */
	private double updateWeight(final int index) {
		if (Math.abs(this.gradients[index]) < this.zeroTolerance) {
			return 0;
		} else if (this.gradients[index] > 0) {
			return this.learningRate;
		} else {
			return -this.learningRate;
		}
	}

}