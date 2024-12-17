from typing import Final, Union, Optional, Iterable, List, Tuple
import igraph
import torch
from einops.layers.torch import Rearrange
from pytorch_symbolic import SymbolicModel, Input, useful_layers as sl, add_to_graph
from pytorch_symbolic.symbolic_data import SymbolicTensor
from models.layers.attention_utils import CustomMultiHeadAttention
from models.tensornet import TensorNetwork, MULT_SKIP_CONN, PROCESSING_ORDER_DEFAULT, SENSOR, MOTOR, SENSOR_MOTOR, \
    LinearNodeTensorUnit, ADD_SKIP_CONN, INTERNEURON, SENSOR_INTERNEURON, INTERNEURON_MOTOR, INTERNEURON_SENSOR, \
    MOTOR_SENSOR, MOTOR_INTERNEURON, SENSOR_SENSOR, INTERNEURON_INTERNEURON, MOTOR_MOTOR
from models.tensornet.tensornet import HEAD_DIM, CROSS_ATTN, SELF_CROSS_ATTN, CROSS_ATTN_SKIP_CONN, SELF_ATTN
from models.layers.misc import nans_like
from utils.connectome_reader import ConnectomeReader, INTERNEURON_ROLE_ORIGINAL


MIX_INPUT_SEQ: Final[str] = "mix"
INDEPENDENT_INPUT_SEQ: Final[str] = "independent"
RECURRENT_INPUT_SEQ: Final[str] = "recurrent"
CONCAT_RECURRENCE: Final[str] = "concat"
LEARNABLEQ_CROSS_ATTN: Final[str] = "learn-q-cross"
LEARNABLEQ_CROSS_SKIP_ATTN: Final[str] = "learn-q-cross-skip"
LEARNABLEQ_SELF_CROSS_ATTN: Final[str] = "learn-q-self-cross"
NO_ATTN_AVG: Final[str] = "no-attn-avg"
NO_ATTN_SUM: Final[str] = "no-attn-sum"
NO_ATTN_MAX: Final[str] = "no-attn-max"
NO_ATTN_LEARNABLEQ: Final[str] = "no-attn-learn-q"
NO_ATTN_LSTM: Final[str] = "no-attn-lstm"
NO_ATT_ROUTER: Final[str] = "no-attn-router"
AVG_RECURRENCE: Final[str] = "avg"
ADD_RECURRENCE: Final[str] = "add"
MAX_RECURRENCE: Final[str] = "max"
LAST_STEP_RECURRENCE: Final[str] = "last"
_MAX_DFS_RECURRENCE_DEPTH: Final[int] = 4


class FunctionalTensorNetwork(TensorNetwork):
    def __init__(self,
                 source: Union[str, ConnectomeReader],
                 input_dim: int,
                 embedding_dim: int,
                 activation: str = "gelu",
                 n_heads: Optional[int] = None,
                 head_dim: Optional[int] = HEAD_DIM,
                 dropout_mha: float = 0.0,
                 attention_type: str = CROSS_ATTN,
                 norm_output: bool = True,
                 norm_tensor_unit: bool = False,
                 skip_connection_type: str = MULT_SKIP_CONN,
                 reapply_unit_synapses: Optional[Iterable[str]] = None,
                 skip_conn_only_synapses: Optional[Iterable[str]] = None,
                 ignored_synapses: Optional[Iterable[str]] = None,
                 processing_order: Optional[str] = PROCESSING_ORDER_DEFAULT,
                 recurrence_type: Optional[str] = None,
                 input_sequences: Optional[str] = None,
                 verbose: bool = True,
                 device: Optional[Union[str, torch.device]] = None):
        """
        The TensorNetwork module designed to emulate the structure and topology of natural connectomes. The model
        consists of interconnected sensor, interneuron, and motor neurons, with heterophilic (between different neuron
        types) and homophilic (between the same neuron types) connections. This functional implementation employs the
        PyTorch Symbolic API (similar to Keras Functional API) to make the complex computation graphs used in the
        TensorNetwork more efficient and automatically parallel.

        :param source: either a string representing the filename of a connectome file or a ConnectomeReader object that
            has already read the connectome file.
        :type source: Union[str, ConnectomeReader]
        :param input_dim: the dimensionality of the input features for each neuron.
        :type input_dim: int
        :param embedding_dim: the dimensionality of the embedding space. In this code, it is used to set the dimensions
            of the Multi-Head Attention layer (if head_dim is not given) and the NodeTensorUnit layers representing the
            neurons.
        :type embedding_dim: int
        :param activation: activation function to use for each dense layer ("gelu" by default).
        :type activation: str
        :param n_heads: the number of attention heads to be used in the Multi-Head Attention layer. It is a
            hyperparameter that determines how many different ways the model can attend to different parts of the input
            sequence simultaneously. Increasing the number of heads can improve the model's ability to capture complex
            patterns in the data. By default, it uses as many heads as the number of motor neurons.
        :type n_heads: Optional[int]
        :param head_dim: the dimension of the key/query/value vector of a single head the multi-head attention layer.
            The multi-head attention embedding dimension will be num_heads*head_dim, and more projections will be added
            in order to manage the embedding sizes. By default, this is HEAD_DIM.
        :type head_dim: Optional[int]
        :param dropout_mha: a float value representing the dropout probability for the Multi-Head Attention layer. It
            determines the probability of an element being zeroed out during training to prevent overfitting.
        :type dropout_mha: float
        :param attention_type: a string indicating whether the multi-head attention module should use the input as
            a query, performing cross-attention/cross-attention followed by a skip-connection with the input("cross"/
            "cross-skip") between motor outputs and the input, or between the motor outputs themselves "self"
            performing a self-attention operation, or both the types "self-cross". By default, this is set to "cross".
        :type attention_type: str
        :param norm_output: whether to apply layer normalization to output.
        :type norm_output: bool
        :param norm_tensor_unit: whether to apply layer normalization to output of each neuron. WARNING: setting this to
            False when using "mult" as skip-connection mechanism can result in Inf, -Inf or NaN output values due to the
            high number of element-wise multiplications. Default is True.
        :type norm_output: bool
        :param skip_connection_type: the type of skip-connection to use, either "add" or "mult"
        :type skip_connection_type: str
        :param reapply_unit_synapses: types of connections that are always implemented by giving the output of the first
            neuron in input to the linear layer of the second neuron, and never only as skip-connections in the
            network, even if the output of the target neuron is already defined; either "SS" (sensor to sensor), "SI"
            (sensor to interneuron), "SM" (sensor to motor), "IS" (interneuron to sensor), "II" (interneuron to
            interneuron), "IM" (interneuron to motor), "MS" (motor to sensor), "MI" (motor to interneuron) or "MM"
            (motor to motor)
        :type reapply_unit_synapses: Iterable[str]
        :param skip_conn_only_synapses: types of connections that are implemented only as skip-connections in the
            network, and not by giving the output of the first neuron in input to the linear layer of the second neuron;
            either "SS" (sensor to sensor), "SI" (sensor to interneuron), "SM" (sensor to motor), "IS" (interneuron to
            sensor), "II" (interneuron to interneuron), "IM" (interneuron to motor), "MS" (motor to sensor), "MI" (motor
            to interneuron) or "MM" (motor to motor)
        :type skip_conn_only_synapses: Iterable[str]
        :param ignored_synapses: types of synapses to ignore; either "SS" (sensor to sensor), "SI" (sensor to
            interneuron), "SM" (sensor to motor), "IS" (interneuron to sensor), "II" (interneuron to interneuron),
            "IM" (interneuron to motor), "MS" (motor to sensor), "MI" (motor to interneuron) or "MM" (motor to motor)
        :type ignored_synapses: Iterable[str]
        :param processing_order: a string representing the order in which the type of connections between neurons are
            processed, separated by comma ("S->O" stands for sensor to others, "S->S" for sensor to sensor, "S->I" for
            sensor to interneuron, "S->M" for sensor to motor, "I->O" for interneuron to others, "I->S" for interneuron
            to sensor, "I->I" for interneuron to interneuron, "I->M" for interneuron to motor, "M->O" for motor to
            others, "M->S" for motor to sensor, "M->I" for motor to interneuron, and "M->M" for motor to motor). Default
            is "S->O,I->O,M->O,S->S,I->I,M->M").
        :type processing_order: str
        :param recurrence_type: type of recurrence to use (either None, "add", "avg", "max", "last" or "concat"). With
            None recurrence is disabled, with "add" the outputs from each timestep will be summed, with "avg" they will
            be averaged, with "max" they will be aggregated through max-over-time, and with "last" only the last output
            will be used. Default is None.
        :type recurrence_type: Optional[str]
        :param input_sequences: whether to input sequence tensors with shape (batch_size, seq_len, embedding_dim), if
            not None, instead of a single batched tensor with shape (batch_size, embedding_dim), with None. If set to
            "independent", the tensornet will be applied completely independently to each timestep, while it set to
            "mix" the tensornet will be still applied independently to each timestep, but in the cross attention
            operation ech query will attend to motor outputs coming from other timesteps, mixing the timesteps in the
            attention operation.If recurrence is enabled (recurrence_type != None), this is automatically set to
            "recurrent". Default is None.
        :type input_sequences: bool
        :param verbose: whether to log the operations during the functional network allocation. Default is True.
        :type verbose: bool
        :param device: The `device` parameter is an optional argument that specifies the device on which the
            computations will be performed. It can be a string specifying the device (e.g. "cpu" or "cuda"), or a
            torch.device object. If not specified, the default device will be used.
        :type device: Optional[Union[str, torch.device]]
        """
        super().__init__(
            source=source,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            activation=activation,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout_mha=dropout_mha,
            attention_type=attention_type,
            norm_output=norm_output,
            norm_tensor_unit=norm_tensor_unit,
            skip_connection_type=skip_connection_type,
            reapply_unit_synapses=reapply_unit_synapses,
            skip_conn_only_synapses=skip_conn_only_synapses,
            ignored_synapses=ignored_synapses,
            processing_order=processing_order,
            device=device
        )
        # Clear unwanted layers (as they are going to be created in a functional way by the "_build network()" method)
        del self._mha, self._layer_norm, self._sensor_neurons, self._interneurons, self._motor_neurons

        # Normal dictionaries so the layers within will not be registered (PyTorch Symbolic API will take care of that)
        self._sensor_neurons = {}
        self._interneurons = {}
        self._motor_neurons = {}
        self._mha = None
        self._mha2 = None

        # Setup instance variables
        self._head_dim: int = head_dim
        self._n_heads: Optional[int] = n_heads
        self._recurrrence_type: Optional[str] = recurrence_type

        self._input_sequences = input_sequences
        if recurrence_type is not None:
            self._input_sequences: Optional[str] = RECURRENT_INPUT_SEQ   # True if recurrence is enabled

        # Learnable query of the cross-attention operation, if needed
        if attention_type in [LEARNABLEQ_CROSS_ATTN, LEARNABLEQ_CROSS_SKIP_ATTN, LEARNABLEQ_SELF_CROSS_ATTN,
                              NO_ATTN_LEARNABLEQ]:
            self._query = torch.nn.Parameter(torch.randn(embedding_dim), requires_grad=True)

        # Build the functional tensor network
        self._functional_network = self._build_functional_network(verbose=verbose)

        # Build also the functional recurrent network if required
        if recurrence_type is not None:
            self._recurrent_functional_networks = self._build_recurrent_functional_network(verbose=verbose)

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def n_heads(self) -> int:
        return self._n_heads

    @property
    def recurrrence_type(self) -> Optional[str]:
        return self._recurrrence_type

    @property
    def input_sequences(self) -> Optional[str]:
        return self._input_sequences

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        # Get parent class constructor params
        constructor_params = super().serialize_constructor_params()

        # Delete unused constructor parameters
        if "adjust_n_heads" in constructor_params:
            del constructor_params["adjust_n_heads"]

        # Add subclass-specific constructor params
        constructor_params["recurrrence_type"] = self._recurrrence_type
        constructor_params["input_sequences"] = self._input_sequences

        return constructor_params

    @staticmethod
    def _replace_none_with_nans(tensors: List[Optional[SymbolicTensor]]) -> List[SymbolicTensor]:
        # Get the first not None tensor
        not_none_tensor = None
        for t in tensors:
            if t is not None:
                not_none_tensor = t
                break

        if not_none_tensor is None:
            raise ValueError("No tensor found, only None entries.")

        return [t if t is not None else add_to_graph(nans_like, not_none_tensor) for t in tensors]

    @staticmethod
    def _replace_nans_with_none(tensor: torch.Tensor) -> Tuple[List[Optional[torch.Tensor]], int]:
        # Split on the size 1 (corresponding to the neuron assuming there is a batch dimension)
        splitted_tensor = tensor.split(1, dim=1)

        # Filter the tensors excluding the ones containing nans and replacing them with None
        splitted_tensor = [t if not t.isnan().any() else None for t in splitted_tensor]
        nan_found: int = len([n for n in splitted_tensor if n is None])

        return splitted_tensor, nan_found

    @staticmethod
    def _remove_nans(tensor: torch.Tensor, squeeze: bool = False) -> Tuple[List[torch.Tensor], int]:
        # Split on the size 1 (corresponding to the neuron assuming there is a batch dimension)
        splitted_tensor = tensor.split(1, dim=1)

        # Filter the tensors excluding the ones containing nans and replacing them with None
        splitted_tensor = [t for t in splitted_tensor if not t.isnan().any()]
        nan_found: int = tensor.shape[1] - len(splitted_tensor)

        # Squeeze the tensors on dimension 1 if required
        if squeeze:
            splitted_tensor = [t.squeeze(dim=1) for t in splitted_tensor]

        return splitted_tensor, nan_found

    def _build_functional_network(self, verbose: bool = True) -> SymbolicModel:

        # Define the input for the functional first-step network
        if self.input_sequences is not None and self.recurrrence_type is None:
            # Shape (batch_size, seq_len, embedding_dim) if self.input_sequences is True and recurrence is disabled
            inp = Input(shape=(2, self.input_dim,))  # not actually 2, but there isn't a way to define dynamic shape
        else:
            # Shape (batch_size, embedding_dim) if self.input_sequences is False or recurrence is enabled
            inp = Input(shape=(self.input_dim,))
        inp_pre_projection = inp

        # Add projection if needed
        if self.input_dim != self.embedding_dim:
            inp = self._projection(inp)

        # If recurrence is enabled, then we need also to return sensors, interneurons and motors
        if self.recurrrence_type is not None:
            out, sensors, interneurons, motors = self._elegans_latent_graph_total(
                inp,
                return_all_neurons_output=True,
                verbose=verbose
            )
            return SymbolicModel(inputs=[inp_pre_projection], outputs=[out, sensors, interneurons, motors])
        else:
            out = self._elegans_latent_graph_total(
                inp,
                return_all_neurons_output=False,
                verbose=verbose
            )
            return SymbolicModel(inputs=[inp_pre_projection], outputs=[out])

    def _build_recurrent_functional_network(self, verbose: bool = True) -> torch.nn.ModuleList:
        recurrent_step_networks = torch.nn.ModuleList()
        not_reached_neuron_flag = True
        timesteps = 0
        sensors, interneurons, motors = None, None, None

        # While there are neurons that aren't reachable
        while not_reached_neuron_flag and timesteps < _MAX_DFS_RECURRENCE_DEPTH:

            if verbose:
                print(f"Building recurrent network for timestep {timesteps}...")

            # Mock input to get the shapes and the reached neurons
            mock_input = torch.randn(2, self.input_dim)

            # Call the functional network at the first timestep, and call the last created recurrent net otherwise
            if timesteps == 0:
                out, sensors, interneurons, motors = self._functional_network(mock_input)
            else:
                # Parse the previous iteration sensors, interneurons, motors lists to remove None entries
                sensors = [s.squeeze(dim=1) for s in sensors if s is not None]
                interneurons = [i.squeeze(dim=1) for i in interneurons if i is not None]
                motors = [m.squeeze(dim=1) for m in motors if m is not None]
                net = recurrent_step_networks[-1]

                # Get the outputs of the recurrent net
                out, sensors, interneurons, motors = net(mock_input, *sensors, *interneurons, *motors)

            # Get output shapes from the tensornet outputs (excluding the batch and neuron index dims)
            sensor_out_shape = tuple(sensors.shape[2:])
            interneuron_out_shape = tuple(interneurons.shape[2:])
            motor_out_shape = tuple(motors.shape[2:])

            # Replace any nan neuron output tensor with None (because it means the neuron has not been reached in that
            # specific list of activations)
            sensors, nan_found_sensors = self._replace_nans_with_none(sensors)
            interneurons, nan_found_interneurons = self._replace_nans_with_none(interneurons)
            motors, nan_found_motors = self._replace_nans_with_none(motors)

            # Count how much nan-filled tensors were found, and if a neuron wasn't reached, then set the flag to true
            nan_found_total = nan_found_sensors + nan_found_interneurons + nan_found_motors
            reached_neurons = self.connectome_graph.vcount()*3 - nan_found_total
            not_reached_neuron_flag = reached_neurons < self.connectome_graph.vcount()

            # Replace real tensors with symbolic inputs
            sensors_symb = [Input(shape=sensor_out_shape) if s is not None else None for s in sensors]
            interneurons_symb = [Input(shape=interneuron_out_shape) if i is not None else None for i in interneurons]
            motors_symb = [Input(shape=motor_out_shape) if m is not None else None for m in motors]

            # Define the inputs (since it is the recurrent network, sensors, interneurons and motors are also needed)
            inp_symb = Input(shape=(self.input_dim,))
            inp_pre_proj_symb = inp_symb
            sensors_inp_symb = [s for s in sensors_symb if s is not None]
            interneurons_inp_symb = [i for i in interneurons_symb if i is not None]
            motors_inp_symb = [m for m in motors_symb if m is not None]

            # Add projection if needed
            if self.input_dim != self.embedding_dim:
                inp_symb = self._projection(inp_symb)

            # Define the network of the timestep as a symbolic model (all timesteps share the same weights)
            symb_out, symb_sensors_out, symb_interneurons_out, symb_motors_out = self._elegans_latent_graph_total(
                symbolic_input=inp_symb,
                sensors=sensors_symb,
                interneurons=interneurons_symb,
                motors=motors_symb,
                return_all_neurons_output=True,
                verbose=verbose
            )

            # TODO: check if it uses the already allocated layers (i.e. correct weight-sharing across timesteps)
            model = SymbolicModel(
                inputs=[inp_pre_proj_symb, *sensors_inp_symb, *interneurons_inp_symb, *motors_inp_symb],
                outputs=[symb_out, symb_sensors_out, symb_interneurons_out, symb_motors_out]
            )
            recurrent_step_networks.append(model)

            # Increase the timestep counter
            timesteps += 1

        return recurrent_step_networks

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        if self.recurrrence_type is not None:

            if len(x.shape) < 3:
                raise ValueError(f"Input tensor should be at least 3-dimensional. "
                                 f"{len(x.shape)}-dimensional tensor given")

            # Split the input on the sequence length dimension (1)
            x = [t.squeeze(dim=1) for t in x.split(1, dim=1)]

            # Output list
            outputs = []

            # For each timestep
            sensors, interneurons, motors = [], [], []
            for i, t in enumerate(x):

                # At the first timestep apply the regular functional network
                if i == 0:
                    out, sensors, interneurons, motors = self._functional_network(t)

                # Otherwise apply recurrence with the i-th recurrent functional network (all share same weights but
                # possibly different topology)
                else:
                    net = self._recurrent_functional_networks[i if i < len(self._recurrent_functional_networks) else -1]

                    # Parse sensors, interneurons, motors to remove nans
                    sensors, _ = self._remove_nans(sensors, squeeze=True)
                    interneurons, _ = self._remove_nans(interneurons, squeeze=True)
                    motors, _ = self._remove_nans(motors, squeeze=True)

                    # Get the output
                    out, sensors, interneurons, motors = net(t, *sensors, *interneurons, *motors)

                # Add the output to the outputs
                outputs.append(out)

            # Aggregate the outputs
            if self.recurrrence_type == LAST_STEP_RECURRENCE:
                return outputs[-1]
            elif self.recurrrence_type == MAX_RECURRENCE:
                outputs = sl.StackLayer(dim=1)(*outputs)
                return outputs.max(dim=1)[0]  # max-over-time
            elif self.recurrrence_type == ADD_RECURRENCE:
                outputs = sl.StackLayer(dim=1)(*outputs)
                return outputs.sum(dim=1)
            elif self.recurrrence_type == CONCAT_RECURRENCE:
                # TODO: implement this
                raise NotImplementedError(f"Recurrence type {CONCAT_RECURRENCE} not implemented yet.")
            else:
                outputs = sl.StackLayer(dim=1)(*outputs)
                return outputs.mean(dim=1)

        return self._functional_network(x)

    def _sensors_to_others(self,
                           sensors: List[Optional[Union[Input, SymbolicTensor]]],
                           interneurons: List[Optional[SymbolicTensor]],
                           motors: List[Optional[SymbolicTensor]],
                           sens_to_inter: bool = True,
                           sens_to_mot: bool = True,
                           verbose: bool = True) -> Tuple[List[Optional[SymbolicTensor]],
                                                          List[Optional[SymbolicTensor]],
                                                          List[Optional[SymbolicTensor]]]:

        # Scan the adjacency matrix to in search of connections from sensors to others

        if verbose:
            print("S to I and M")

        for i in range(0, self.connectome_graph.vcount()):
            for j in range(0, self.connectome_graph.vcount()):
                try:

                    # Check if the i->j edge exists
                    coord_0 = int(i)
                    coord_1 = int(j)
                    self.connectome_graph.get_eid(coord_0, coord_1)  # will throw error if edge doesn't exist

                    # If the i output tensor is defined
                    if sensors[coord_0] is not None:

                        # If i is a sensor, j is a motor and sensor->motor edges are not to be ignored
                        if self.connectome_graph.vs()['role'][coord_0] in [SENSOR]:
                            if self.connectome_graph.vs()['role'][coord_1] in [MOTOR] \
                                    and SENSOR_MOTOR not in self.ignored_synapses and sens_to_mot:

                                # If the j output tensor is undefined
                                if motors[coord_1] is None:

                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._motor_neurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=MOTOR)
                                        if self.norm_tensor_unit:
                                            self._motor_neurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._motor_neurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._motor_neurons[str(coord_1)]

                                    # Apply the j layer and the activation function
                                    motors[coord_1] = neuron_layer(sensors[coord_0])
                                    motors[coord_1] = self._activation_fn(motors[coord_1])

                                # Otherwise, if the j output is already defined
                                else:

                                    # Get the j neuron layer
                                    neuron_layer = self._motor_neurons[str(coord_1)]

                                    # If required, apply the neuron layer again and add the additive skip-connection
                                    if SENSOR_MOTOR in self.reapply_unit_synapses:
                                        tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                                        motors[coord_1] = \
                                            self._activation_fn(tensor_unit(sensors[coord_0])) + motors[coord_1]

                                    # Otherwise just apply the required type of skip-connection
                                    elif self.skip_connection_type == ADD_SKIP_CONN:
                                        motors[coord_1] = sensors[coord_0] + motors[coord_1]
                                    else:  # MULT_SKIP_CONN
                                        motors[coord_1] = sensors[coord_0] * motors[coord_1]

                                    # Normalize if required
                                    if self.norm_tensor_unit:
                                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                                        motors[coord_1] = layer_norm(motors[coord_1])

                            # If i is a sensor, j is an interneuron and sensor->interneuron edges are not to be ignored
                            if self.connectome_graph.vs()['role'][coord_1] in [INTERNEURON, INTERNEURON_ROLE_ORIGINAL] \
                                    and SENSOR_INTERNEURON not in self.ignored_synapses and sens_to_inter:

                                # If the j output tensor is undefined
                                if interneurons[coord_1] is None:

                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._interneurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=INTERNEURON)
                                        if self.norm_tensor_unit:
                                            self._interneurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._interneurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._interneurons[str(coord_1)]

                                    # Apply the j layer and the activation function
                                    interneurons[coord_1] = neuron_layer(sensors[coord_0])
                                    interneurons[coord_1] = self._activation_fn(interneurons[coord_1])

                                # Otherwise, if the j output is already defined
                                else:

                                    # Get the j neuron layer
                                    neuron_layer = self._interneurons[str(coord_1)]

                                    # If required, apply the neuron layer again and add the additive skip-connection
                                    if SENSOR_INTERNEURON in self.reapply_unit_synapses:
                                        tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                                        interneurons[coord_1] = \
                                            self._activation_fn(tensor_unit(sensors[coord_0])) + interneurons[coord_1]

                                    # Otherwise just apply the required type of skip-connection
                                    elif self.skip_connection_type == ADD_SKIP_CONN:
                                        interneurons[coord_1] = sensors[coord_0] + interneurons[coord_1]
                                    else:  # MULT_SKIP_CONN
                                        interneurons[coord_1] = sensors[coord_0] * interneurons[coord_1]

                                    # Normalize if required
                                    if self.norm_tensor_unit:
                                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                                        interneurons[coord_1] = layer_norm(interneurons[coord_1])

                except igraph.InternalError:
                    continue

        return sensors, interneurons, motors

    def _interneurons_to_others(self,
                                sensors: List[Optional[Union[Input, SymbolicTensor]]],
                                interneurons: List[Optional[SymbolicTensor]],
                                motors: List[Optional[SymbolicTensor]],
                                int_to_sens: bool = True,
                                int_to_mot: bool = True,
                                verbose: bool = True) -> Tuple[List[Optional[SymbolicTensor]],
                                                               List[Optional[SymbolicTensor]],
                                                               List[Optional[SymbolicTensor]]]:

        # Scan the adjacency matrix to in search of connections from interneurons to others
        if verbose:
            print("I to S and M")

        for i in range(0, self.connectome_graph.vcount()):
            for j in range(0, self.connectome_graph.vcount()):
                try:

                    # Check if the i->j edge exists
                    coord_0 = int(i)
                    coord_1 = int(j)
                    self.connectome_graph.get_eid(coord_0, coord_1)  # will throw error if edge doesn't exist

                    # If the i output tensor is defined
                    if interneurons[coord_0] is not None:

                        # If i is an interneuron, j is a motor and interneuron->motor edges are not to be ignored
                        if self.connectome_graph.vs()['role'][coord_0] in [INTERNEURON]:
                            if self.connectome_graph.vs()['role'][coord_1] in [MOTOR] \
                                    and INTERNEURON_MOTOR not in self.ignored_synapses and int_to_mot:

                                # If the j output tensor is undefined
                                if motors[coord_1] is None:

                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._motor_neurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=MOTOR)
                                        if self.norm_tensor_unit:
                                            self._motor_neurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._motor_neurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._motor_neurons[str(coord_1)]

                                    # Apply the j layer and the activation function
                                    motors[coord_1] = neuron_layer(interneurons[coord_0])
                                    motors[coord_1] = self._activation_fn(motors[coord_1])

                                # Otherwise if the j output is already defined
                                else:
                                    # Get the j neuron layer
                                    neuron_layer = self._motor_neurons[str(coord_1)]

                                    # If required, apply the neuron layer again and add the additive skip-connection
                                    if INTERNEURON_MOTOR in self.reapply_unit_synapses:
                                        tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                                        motors[coord_1] = \
                                            self._activation_fn(tensor_unit(interneurons[coord_0])) + motors[coord_1]

                                    # Otherwise just apply the required type of skip-connection
                                    elif self.skip_connection_type == ADD_SKIP_CONN:
                                        motors[coord_1] = interneurons[coord_0] + motors[coord_1]
                                    else:  # MULT_SKIP_CONN
                                        motors[coord_1] = interneurons[coord_0] * motors[coord_1]

                                    # Normalize if required
                                    if self.norm_tensor_unit:
                                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                                        motors[coord_1] = layer_norm(motors[coord_1])

                            # If i is an interneuron, j is a sensor and interneuron->sensor edges are not to be ignored
                            if self.connectome_graph.vs()['role'][coord_1] in [SENSOR] \
                                    and INTERNEURON_SENSOR not in self.ignored_synapses and int_to_sens:

                                # If the j output tensor is undefined
                                if sensors[coord_1] is None:

                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._sensor_neurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=SENSOR)
                                        if self.norm_tensor_unit:
                                            self._sensor_neurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._sensor_neurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._sensor_neurons[str(coord_1)]

                                    # Apply the j neuron layer
                                    sensors[coord_1] = neuron_layer(interneurons[coord_0])
                                    sensors[coord_1] = self._activation_fn(sensors[coord_1])

                                # Otherwise if the j output tensor is already defined
                                else:
                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._sensor_neurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=SENSOR)
                                        if self.norm_tensor_unit:
                                            self._sensor_neurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._sensor_neurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._sensor_neurons[str(coord_1)]

                                    # If required, apply the neuron layer again and add the additive skip-connection
                                    if INTERNEURON_SENSOR in self.reapply_unit_synapses:
                                        tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                                        sensors[coord_1] = \
                                            self._activation_fn(tensor_unit(interneurons[coord_0])) + sensors[coord_1]

                                    # Otherwise just apply the required type of skip-connection
                                    elif self.skip_connection_type == ADD_SKIP_CONN:
                                        sensors[coord_1] = interneurons[coord_0] + sensors[coord_1]
                                    else:  # MULT_SKIP_CONN
                                        sensors[coord_1] = interneurons[coord_0] * sensors[coord_1]

                                    # Normalize if required
                                    if self.norm_tensor_unit:
                                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                                        sensors[coord_1] = layer_norm(sensors[coord_1])

                except igraph.InternalError:
                    continue

        return sensors, interneurons, motors

    def _motors_to_others(self,
                          sensors: List[Optional[Union[Input, SymbolicTensor]]],
                          interneurons: List[Optional[SymbolicTensor]],
                          motors: List[Optional[SymbolicTensor]],
                          mot_to_sens: bool = True,
                          mot_to_int: bool = True,
                          verbose: bool = True) -> Tuple[List[Optional[SymbolicTensor]],
                                                         List[Optional[SymbolicTensor]],
                                                         List[Optional[SymbolicTensor]]]:

        # Scan the adjacency matrix searching for motors to other connections
        if verbose:
            print("M to I and S")

        for i in range(0, self.connectome_graph.vcount()):
            for j in range(0, self.connectome_graph.vcount()):
                try:

                    # Check if the i->j edge exists
                    coord_0 = int(i)
                    coord_1 = int(j)
                    self.connectome_graph.get_eid(coord_0, coord_1)  # will throw error if edge doesn't exist

                    # If the i output tensor is defined
                    if motors[coord_0] is not None:

                        # If i is a motor
                        if self.connectome_graph.vs()['role'][coord_0] in [MOTOR]:

                            # If j is a sensor and motor->sensor edges are not to be ignored
                            if self.connectome_graph.vs()['role'][coord_1] in [SENSOR] \
                                    and MOTOR_SENSOR not in self.ignored_synapses and mot_to_sens:

                                # If the j output tensor is undefined
                                if sensors[coord_1] is None:

                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._sensor_neurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=SENSOR)
                                        if self.norm_tensor_unit:
                                            self._sensor_neurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._sensor_neurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._sensor_neurons[str(coord_1)]

                                    # Apply the j neuron layer and activation function
                                    sensors[coord_1] = neuron_layer(motors[coord_0])
                                    sensors[coord_1] = self._activation_fn(sensors[coord_1])

                                # Otherwise if the j output tensor is already defined
                                else:
                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._sensor_neurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=SENSOR)
                                        if self.norm_tensor_unit:
                                            self._sensor_neurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._sensor_neurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._sensor_neurons[str(coord_1)]

                                    # If required, apply the neuron layer again and add the additive skip-connection
                                    if MOTOR_SENSOR in self.reapply_unit_synapses:
                                        tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                                        sensors[coord_1] = \
                                            self._activation_fn(tensor_unit(motors[coord_0])) + sensors[coord_1]

                                    # Otherwise just apply the required type of skip-connection
                                    elif self.skip_connection_type == ADD_SKIP_CONN:
                                        sensors[coord_1] = motors[coord_0] + sensors[coord_1]
                                    else:  # MULT_SKIP_CONN
                                        sensors[coord_1] = motors[coord_0] * sensors[coord_1]

                                    # Normalize if required
                                    if self.norm_tensor_unit:
                                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                                        sensors[coord_1] = layer_norm(sensors[coord_1])

                            # If j is an interneuron and motor->interneuron edges are not to be ignored
                            if self.connectome_graph.vs()['role'][coord_1] in [INTERNEURON, INTERNEURON_ROLE_ORIGINAL] \
                                    and MOTOR_INTERNEURON not in self.ignored_synapses and mot_to_int:

                                # If the j output tensor is undefined
                                if interneurons[coord_1] is None:

                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._interneurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=INTERNEURON)
                                        if self.norm_tensor_unit:
                                            self._interneurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._interneurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._interneurons[str(coord_1)]

                                    # Apply the j neuron layer and activation function
                                    interneurons[coord_1] = neuron_layer(motors[coord_0])
                                    interneurons[coord_1] = self._activation_fn(interneurons[coord_1])

                                # Otherwise if the j output tensor is already defined
                                else:
                                    # Get the j neuron layer
                                    neuron_layer = self._interneurons[str(coord_1)]

                                    # If required, apply the neuron layer again and add the additive skip-connection
                                    if MOTOR_INTERNEURON in self.reapply_unit_synapses:
                                        tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                                        interneurons[coord_1] = \
                                            self._activation_fn(tensor_unit(motors[coord_0])) + interneurons[coord_1]

                                    # Otherwise just apply the required type of skip-connection
                                    elif self.skip_connection_type == ADD_SKIP_CONN:
                                        interneurons[coord_1] = motors[coord_0] + interneurons[coord_1]
                                    else:  # MULT_SKIP_CONN
                                        interneurons[coord_1] = motors[coord_0] * interneurons[coord_1]

                                    # Normalize if required
                                    if self.norm_tensor_unit:
                                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                                        interneurons[coord_1] = layer_norm(interneurons[coord_1])

                except igraph.InternalError:
                    continue

        return sensors, interneurons, motors

    def _sensors_to_sensors(self,
                            sensors: List[Optional[SymbolicTensor]],
                            verbose: bool = True) -> List[Optional[SymbolicTensor]]:

        # Scan the adjacency matrix in search of sensor to sensor connections
        if verbose:
            print("S to S")

        for i in range(0, self.connectome_graph.vcount()):
            for j in range(0, self.connectome_graph.vcount()):
                try:

                    # Check if the i->j edge exists
                    coord_0 = int(i)
                    coord_1 = int(j)
                    self.connectome_graph.get_eid(coord_0, coord_1)  # will throw error if edge doesn't exist

                    # If the i output tensor is defined
                    if sensors[coord_0] is not None:

                        # If i and j are sensors and sensor->sensor edges are not to be ignored
                        if self.connectome_graph.vs()['role'][coord_0] in [SENSOR]:
                            if self.connectome_graph.vs()['role'][coord_1] in [SENSOR] \
                                    and SENSOR_SENSOR not in self.ignored_synapses:

                                # If the j output tensor is undefined
                                if sensors[coord_1] is None:

                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._sensor_neurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=SENSOR)
                                        if self.norm_tensor_unit:
                                            self._sensor_neurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._sensor_neurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._sensor_neurons[str(coord_1)]

                                    # Apply the j neuron layer and activation function
                                    sensors[coord_1] = neuron_layer(sensors[coord_0])
                                    sensors[coord_1] = self._activation_fn(sensors[coord_1])

                                # Otherwise if the j output tensor is already defined
                                else:
                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._sensor_neurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=SENSOR)
                                        if self.norm_tensor_unit:
                                            self._sensor_neurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._sensor_neurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._sensor_neurons[str(coord_1)]

                                    # If required, apply the neuron layer again and add the additive skip-connection
                                    if SENSOR_SENSOR in self.reapply_unit_synapses:
                                        tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                                        sensors[coord_1] = \
                                            self._activation_fn(tensor_unit(sensors[coord_0])) + sensors[coord_1]

                                    # Otherwise just apply the required type of skip-connection
                                    elif self.skip_connection_type == ADD_SKIP_CONN:
                                        sensors[coord_1] = sensors[coord_0] + sensors[coord_1]
                                    else:  # MULT_SKIP_CONN
                                        sensors[coord_1] = sensors[coord_0] * sensors[coord_1]

                                    # Normalize if required
                                    if self.norm_tensor_unit:
                                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                                        sensors[coord_1] = layer_norm(sensors[coord_1])

                except igraph.InternalError:
                    continue

        return sensors

    def _interneurons_to_interneurons(self,
                                      interneurons: List[Optional[SymbolicTensor]],
                                      verbose: bool = True) -> List[Optional[SymbolicTensor]]:

        # Scan the adjacency matrix in search of interneuron to interneuron connections
        if verbose:
            print("I to I")

        for i in range(0, self.connectome_graph.vcount()):
            for j in range(0, self.connectome_graph.vcount()):
                try:

                    # Check if the i->j edge exist
                    coord_0 = int(i)
                    coord_1 = int(j)
                    self.connectome_graph.get_eid(coord_0, coord_1)  # will throw error if edge doesn't exist

                    # If the i output tensor is defined
                    if interneurons[coord_0] is not None:

                        # If i and j are interneurons and interneuron->interneuron edges are not to be ignored
                        if self.connectome_graph.vs()['role'][coord_0] in [INTERNEURON, INTERNEURON_ROLE_ORIGINAL]:
                            if self.connectome_graph.vs()['role'][coord_1] in [INTERNEURON, INTERNEURON_ROLE_ORIGINAL] \
                                    and INTERNEURON_INTERNEURON not in self.ignored_synapses:

                                # If the j output tensor is undefined
                                if interneurons[coord_1] is None:

                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._interneurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=INTERNEURON)
                                        if self.norm_tensor_unit:
                                            self._interneurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._interneurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._interneurons[str(coord_1)]

                                    # Apply the j neuron layer and activation function
                                    interneurons[coord_1] = neuron_layer(interneurons[coord_0])
                                    interneurons[coord_1] = self._activation_fn(interneurons[coord_1])

                                # Otherwise if the j output tensor is defined
                                else:
                                    # Get the j neuron layer
                                    neuron_layer = self._interneurons[str(coord_1)]

                                    # If required, apply the neuron layer again and add the additive skip-connection
                                    if INTERNEURON_INTERNEURON in self.reapply_unit_synapses:
                                        tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                                        interneurons[coord_1] = self._activation_fn(
                                            tensor_unit(interneurons[coord_0])
                                        ) + interneurons[coord_1]

                                    # Otherwise just apply the required type of skip-connection
                                    elif self.skip_connection_type == ADD_SKIP_CONN:
                                        interneurons[coord_1] = interneurons[coord_0] + interneurons[coord_1]
                                    else:  # MULT_SKIP_CONN
                                        interneurons[coord_1] = interneurons[coord_0] * interneurons[coord_1]

                                    # Normalize if required
                                    if self.norm_tensor_unit:
                                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                                        interneurons[coord_1] = layer_norm(interneurons[coord_1])

                except igraph.InternalError:
                    continue

        return interneurons

    def _motors_to_motors(self,
                          motors: List[Optional[SymbolicTensor]],
                          verbose: bool = True) -> List[Optional[SymbolicTensor]]:

        # Scan the adjacency matrix in search of motor->motor connections
        if verbose:
            print("M to M")

        for i in range(0, self.connectome_graph.vcount()):
            for j in range(0, self.connectome_graph.vcount()):
                try:

                    # Check if the i->j edge exist
                    coord_0 = int(i)
                    coord_1 = int(j)
                    self.connectome_graph.get_eid(coord_0, coord_1)  # will throw error if edge doesn't exist

                    # If the i output tensor is defined
                    if motors[coord_0] is not None:

                        # If i and j are motors and motor->motor edges are not to be ignored
                        if self.connectome_graph.vs()['role'][coord_0] in [MOTOR]:
                            if self.connectome_graph.vs()['role'][coord_1] in [MOTOR] \
                                    and MOTOR_MOTOR not in self.ignored_synapses:

                                # If j output tensor is undefined
                                if motors[coord_1] is None:

                                    # Either allocate the j neuron layer or get it if it already exists
                                    if str(coord_1) not in self._motor_neurons:
                                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                                           self.embedding_dim,
                                                                           neuron_type=MOTOR)
                                        if self.norm_tensor_unit:
                                            self._motor_neurons[str(coord_1)] = torch.nn.Sequential(
                                                tensor_unit,
                                                torch.nn.LayerNorm(self.embedding_dim)
                                            )
                                        else:
                                            self._motor_neurons[str(coord_1)] = tensor_unit

                                    # Get the j neuron layer
                                    neuron_layer = self._motor_neurons[str(coord_1)]

                                    # Apply the j neuron layer and activation function
                                    motors[coord_1] = neuron_layer(motors[coord_0])
                                    motors[coord_1] = self._activation_fn(motors[coord_1])

                                # Otherwise if the j output tensor is defined
                                else:
                                    # Get the j neuron layer
                                    neuron_layer = self._motor_neurons[str(coord_1)]

                                    # If required, apply the neuron layer again and add the additive skip-connection
                                    if MOTOR_MOTOR in self.reapply_unit_synapses:
                                        tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                                        motors[coord_1] = \
                                            self._activation_fn(tensor_unit(motors[coord_0])) + motors[coord_1]

                                    # Otherwise just apply the required type of skip-connection
                                    elif self.skip_connection_type == ADD_SKIP_CONN:
                                        motors[coord_1] = motors[coord_0] + motors[coord_1]
                                    else:  # MULT_SKIP_CONN
                                        motors[coord_1] = motors[coord_0] * motors[coord_1]

                                    # Normalize if required
                                    if self.norm_tensor_unit:
                                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                                        motors[coord_1] = layer_norm(motors[coord_1])

                except igraph.InternalError:
                    continue

        return motors

    def _init_graph_allocations(self,
                                symbolic_input: Union[Input, SymbolicTensor],
                                sensors: List[Optional[Union[Input, SymbolicTensor]]],
                                interneurons: List[Optional[SymbolicTensor]],
                                motors: List[Optional[SymbolicTensor]],
                                verbose: bool = True) -> Tuple[List[Optional[SymbolicTensor]],
                                                               List[Optional[SymbolicTensor]],
                                                               List[Optional[SymbolicTensor]]]:

        # Initialize the sensor neurons
        sensors = self._init_sensors(symbolic_input=symbolic_input, sensors=sensors, verbose=verbose)

        sensors, interneurons, motors = self._sensors_to_others(
            sensors=sensors,
            interneurons=interneurons,
            motors=motors,
            verbose=verbose
        )
        sensors, interneurons, motors = self._interneurons_to_others(
            sensors=sensors,
            interneurons=interneurons,
            motors=motors,
            verbose=verbose
        )
        sensors, interneurons, motors = self._motors_to_others(
            sensors=sensors,
            interneurons=interneurons,
            motors=motors,
            verbose=verbose
        )

        return sensors, motors, interneurons

    def _init_sensors(self,
                      symbolic_input: Union[Input, SymbolicTensor],
                      sensors: List[Optional[Union[Input, SymbolicTensor]]],
                      verbose: bool = True) -> List[Optional[SymbolicTensor]]:

        if verbose:
            print(f"Starting sensor initialization...")

        # Define the sensor inputs
        role_list = self.connectome_graph.vs()['role']
        for i, role in enumerate(role_list):
            if role in [SENSOR]:

                # If the input of the sensor is undefined, thn just use the symbolic input as sensor output
                if sensors[i] is None:
                    sensors[i] = symbolic_input  # the input

                # Otherwise apply the sensor neuron layer on the input and sum it to the previous state
                else:
                    # Either allocate the sensor neuron layer or get it if it already exists
                    if str(i) not in self._sensor_neurons:
                        tensor_unit = LinearNodeTensorUnit(self.embedding_dim,
                                                           self.embedding_dim,
                                                           neuron_type=SENSOR)
                        if self.norm_tensor_unit:
                            self._sensor_neurons[str(i)] = torch.nn.Sequential(
                                tensor_unit,
                                torch.nn.LayerNorm(self.embedding_dim)
                            )
                        else:
                            self._sensor_neurons[str(i)] = tensor_unit

                    # Get the sensor neuron layer (we are sure that it already exists at this point)
                    neuron_layer = self._sensor_neurons[str(i)]

                    # Apply the neuron layer again and add the additive skip-connection
                    tensor_unit = neuron_layer if not self.norm_tensor_unit else neuron_layer[1]
                    old_sensor_state = sensors[i]
                    sensors[i] = self._activation_fn(tensor_unit(symbolic_input)) + symbolic_input

                    # Normalize if required
                    if self.norm_tensor_unit:
                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                        sensors[i] = layer_norm(sensors[i])

                    # Skip-connection with the old sensor state
                    sensors[i] = sensors[i] + old_sensor_state

                    # Normalize if required
                    if self.norm_tensor_unit:
                        layer_norm = neuron_layer[1]  # the layer norm of this neuron
                        sensors[i] = layer_norm(sensors[i])

        if verbose:
            print("Sensors initialization completed.")

        return sensors

    def _init_graph_inter_allocations(self,
                                      sensors: List[Optional[SymbolicTensor]],
                                      motors: List[Optional[SymbolicTensor]],
                                      interneurons: List[Optional[SymbolicTensor]],
                                      verbose: bool = True) -> Tuple[List[Optional[SymbolicTensor]],
                                                                     List[Optional[SymbolicTensor]],
                                                                     List[Optional[SymbolicTensor]]]:

        sensors = self._sensors_to_sensors(sensors, verbose=verbose)
        interneurons = self._interneurons_to_interneurons(interneurons, verbose=verbose)
        motors = self._motors_to_motors(motors, verbose=verbose)

        return sensors, motors, interneurons

    def _init_processing_ordered_network(self,
                                         symbolic_input: Union[Input, SymbolicTensor],
                                         sensors: List[Optional[Union[Input, SymbolicTensor]]],
                                         interneurons: List[Optional[SymbolicTensor]],
                                         motors: List[Optional[SymbolicTensor]],
                                         verbose: bool = True) -> Tuple[List[Optional[SymbolicTensor]],
                                                                        List[Optional[SymbolicTensor]],
                                                                        List[Optional[SymbolicTensor]]]:

        # Order connections according to the given processing order
        ordered_connection_types = self.processing_order.split(",")

        # Check if sensors are the first in the processing order, otherwise throw an error
        first_two_connection_types = ordered_connection_types[0:2]
        if "S->O" not in first_two_connection_types:
            raise ValueError("'S->O' must be in the first two connection types to process to preserve causality."
                             f"{first_two_connection_types} given.")
        '''if first_two_connection_types[0] != "S->S" and first_two_connection_types[1] != "S->O":
            raise ValueError("The first connection type to process must be either 'S->O' or 'S->S' to preserve "
                             f"causality, '{first_two_connection_types[0]}' given.")'''

        # Initialize sensors
        sensors = self._init_sensors(symbolic_input=symbolic_input, sensors=sensors, verbose=verbose)

        for conn_type in ordered_connection_types:
            if conn_type == "S->S":
                sensors = self._sensors_to_sensors(sensors, verbose=verbose)
            elif conn_type == "S->O":
                sensors, interneurons, motors = self._sensors_to_others(
                    sensors=sensors,
                    interneurons=interneurons,
                    motors=motors,
                    sens_to_inter=True,
                    sens_to_mot=True,
                    verbose=verbose
                )
            elif conn_type == "S->I":
                sensors, interneurons, motors = self._sensors_to_others(
                    sensors=sensors,
                    interneurons=interneurons,
                    motors=motors,
                    sens_to_inter=True,
                    sens_to_mot=False,
                    verbose=verbose
                )
            elif conn_type == "S->M":
                sensors, interneurons, motors = self._sensors_to_others(
                    sensors=sensors,
                    interneurons=interneurons,
                    motors=motors,
                    sens_to_inter=False,
                    sens_to_mot=True,
                    verbose=verbose
                )
            elif conn_type == "I->I":
                interneurons = self._interneurons_to_interneurons(interneurons, verbose=verbose)
            elif conn_type == "I->O":
                sensors, interneurons, motors = self._interneurons_to_others(
                    sensors=sensors,
                    interneurons=interneurons,
                    motors=motors,
                    int_to_sens=True,
                    int_to_mot=True,
                    verbose=verbose
                )
            elif conn_type == "I->S":
                sensors, interneurons, motors = self._interneurons_to_others(
                    sensors=sensors,
                    interneurons=interneurons,
                    motors=motors,
                    int_to_sens=True,
                    int_to_mot=False,
                    verbose=verbose
                )
            elif conn_type == "I->M":
                sensors, interneurons, motors = self._interneurons_to_others(
                    sensors=sensors,
                    interneurons=interneurons,
                    motors=motors,
                    int_to_sens=False,
                    int_to_mot=True,
                    verbose=verbose
                )
            elif conn_type == "M->M":
                motors = self._motors_to_motors(motors, verbose=verbose)
            elif conn_type == "M->O":
                sensors, interneurons, motors = self._motors_to_others(
                    sensors=sensors,
                    interneurons=interneurons,
                    motors=motors,
                    mot_to_sens=True,
                    mot_to_int=True,
                    verbose=verbose
                )
            elif conn_type == "M->S":
                sensors, interneurons, motors = self._motors_to_others(
                    sensors=sensors,
                    interneurons=interneurons,
                    motors=motors,
                    mot_to_sens=True,
                    mot_to_int=False,
                    verbose=verbose
                )
            elif conn_type == "M->I":
                sensors, interneurons, motors = self._motors_to_others(
                    sensors=sensors,
                    interneurons=interneurons,
                    motors=motors,
                    mot_to_sens=False,
                    mot_to_int=True,
                    verbose=verbose
                )

        return sensors, motors, interneurons

    def _elegans_latent_graph_total(self,
                                    symbolic_input: Union[Input, SymbolicTensor],
                                    sensors: List[Optional[Union[Input, SymbolicTensor]]] = None,
                                    interneurons: List[Optional[Union[Input, SymbolicTensor]]] = None,
                                    motors: List[Optional[Union[Input, SymbolicTensor]]] = None,
                                    return_all_neurons_output: bool = False,
                                    verbose: bool = True) -> Union[SymbolicTensor,
                                                                   Tuple[SymbolicTensor,
                                                                         SymbolicTensor,
                                                                         SymbolicTensor,
                                                                         SymbolicTensor]]:
        if sensors is None:
            sensors = [None for _ in range(0, self.connectome_graph.vcount())]

        if interneurons is None:
            interneurons = [None for _ in range(0, self.connectome_graph.vcount())]

        if motors is None:
            motors = [None for _ in range(0, self.connectome_graph.vcount())]

        if verbose:
            print(f"Starting network connections allocation...")

        if self.processing_order == PROCESSING_ORDER_DEFAULT:
            sensors, motors, interneurons = self._init_graph_allocations(symbolic_input=symbolic_input,
                                                                         sensors=sensors,
                                                                         motors=motors,
                                                                         interneurons=interneurons,
                                                                         verbose=verbose)
            sensors, motors, interneurons = self._init_graph_inter_allocations(sensors=sensors,
                                                                               motors=motors,
                                                                               interneurons=interneurons,
                                                                               verbose=verbose)
        else:
            sensors, motors, interneurons = self._init_processing_ordered_network(symbolic_input=symbolic_input,
                                                                                  sensors=sensors,
                                                                                  interneurons=interneurons,
                                                                                  motors=motors,
                                                                                  verbose=verbose)

        # sensors = [item for item in sensors if item is not None]
        # motors = [item for item in motors if item is not None]
        # interneurons = [item for item in interneurons if item is not None]

        # Get the output from motor neurons
        mot_out = [m for m in motors if m is not None]

        non_none_sensors, non_none_interneurons, non_none_motors = None, None, None
        if verbose:
            print(f"Network connections allocation completed.")

            non_none_sensors = len([s for s in sensors if s is not None])
            print(f"Sensors ({non_none_sensors} not None)")
            print(sensors)

            non_none_interneurons = len([i for i in interneurons if i is not None])
            print(f"Interneurons ({non_none_interneurons} not None)")
            print(interneurons)

            non_none_motors = len([m for m in motors if m is not None])
            print(f"Motors ({non_none_motors} not None)")
            print(motors)

            print(f"Stack motor output ({len(mot_out)})")
            print(mot_out)

            print("Starting attention layer allocation...")

        # Stack the (batch_size, embedding_dim) motor tensors into a single (batch_size, n_mot, embedding_dim) tensor
        merged_mot_out = sl.StackLayer(dim=1)(*mot_out)

        # If more than one timestep was provided, with shape (batch_size, n_mot, seq_len, embedding_dim)), convert it
        # into a tensor with shape (batch_size, n_mot*seq_len, embedding_dim) if we want to mix the timesteps, or to
        # a tensor with shape (batch_size*seq_len, n_mot, embedding_dim) if we don't want to mix the timesteps
        n_timesteps = None
        if len(merged_mot_out.shape) > 3:

            if self.input_sequences == INDEPENDENT_INPUT_SEQ:
                n_timesteps = merged_mot_out.size()[2]  # this will be symbolic and will contain the timesteps
                merged_mot_out = Rearrange('b n s e -> (b s) n e')(merged_mot_out)
            else:
                merged_mot_out = Rearrange('b n s e -> b (n s) e')(merged_mot_out)

        # Convert to batch-second (n_mot, batch_size, embedding_dim) or (n_mot*seq_len, batch_size, embedding_dim) or
        # (n_mot, batch_size*seq_len, embedding_dim)
        merged_mot_out = Rearrange('b s e -> s b e')(merged_mot_out)

        if len(symbolic_input.shape) < 3:
            # Unsqueeze on dim=1 because (batch_size, 1, embed_dim) is needed
            symbolic_input = symbolic_input.unsqueeze(dim=1)

        # If input_sequences is not None and we don't want to mix the timesteps, broadcast the input tensor from
        # (batch_size, seq_len, embedding_dim) to (batch_size*seq_len, 1, embedding_dim)
        if self.input_sequences == INDEPENDENT_INPUT_SEQ:
            # Unsqueeze to (batch_size, seq_len, 1, embedding_dim)
            symbolic_input = symbolic_input.unsqueeze(dim=2)

            # Reshape to (batch_size*seq_len, 1, embedding_dim)
            symbolic_input = Rearrange("b s n e -> (b s) n e")(symbolic_input)

        # Convert the input to batch-second (1, batch_size, embedding_dim) or (seq_len, batch_size, embedding_dim)
        symbolic_input = Rearrange('b s e -> s b e')(symbolic_input)

        use_attention = self.attention_type not in [NO_ATTN_LEARNABLEQ, NO_ATTN_AVG, NO_ATTN_MAX, NO_ATTN_SUM,
                                                    NO_ATTN_LSTM, NO_ATT_ROUTER]
        output_tensor = None
        if use_attention:
            if self._mha is None:
                if self._n_heads is None:
                    self._n_heads = len(mot_out)
                # self._n_heads = len(mot_out)  uncomment this if you want to load old models
                self._mha = CustomMultiHeadAttention(
                    embed_dim=self.embedding_dim,
                    head_dim_key=self.head_dim,
                    n_heads=self.n_heads,
                    dropout=self.dropout_mha,
                    batch_first=False,
                    use_with_symbolic_api=True
                )
            if self.attention_type in [SELF_CROSS_ATTN, LEARNABLEQ_SELF_CROSS_ATTN] and self._mha2 is None:
                self._mha2 = CustomMultiHeadAttention(
                    embed_dim=self.embedding_dim,
                    head_dim_key=self.head_dim,
                    n_heads=self.n_heads,
                    dropout=self.dropout_mha,
                    batch_first=False,
                    use_with_symbolic_api=True
                )

            # Perform either cross-attention, self-attention or both according to the parameters
            if self.attention_type == CROSS_ATTN or self.attention_type == CROSS_ATTN_SKIP_CONN:
                output_tensor, _ = self._mha(
                    query=symbolic_input,
                    key=merged_mot_out,
                    value=merged_mot_out,
                    return_attn_weights=True
                )  # cross-attention

                if self.attention_type == CROSS_ATTN_SKIP_CONN:
                    output_tensor = output_tensor + symbolic_input  # skip-connection

            elif self.attention_type == SELF_CROSS_ATTN:
                # Perform self-attention
                x_self, _ = self._mha2(
                    query=merged_mot_out,
                    key=merged_mot_out,
                    value=merged_mot_out
                )  # self-attention
                merged_mot_out = merged_mot_out + x_self  # skip-connection

                # Perform cross-attention
                output_tensor, _ = self._mha(
                    query=symbolic_input,
                    key=merged_mot_out,
                    value=merged_mot_out,
                    return_attn_weights=True
                )  # cross-attention
                output_tensor = output_tensor + symbolic_input  # skip-connection

            elif self.attention_type == LEARNABLEQ_CROSS_ATTN or self.attention_type == LEARNABLEQ_CROSS_SKIP_ATTN:

                # Expand the learnable query to the desired dimension (seq_len, batch_size, embedding_dim)
                q = self._query
                batch_size_symbolic = symbolic_input.size()[1]
                seq_len_symbolic = symbolic_input.size()[0]
                q = add_to_graph(lambda t, s, b: t.expand(s, b, -1), q, seq_len_symbolic, batch_size_symbolic)

                # Apply MHA
                output_tensor, _ = self._mha(
                    query=q,
                    key=merged_mot_out,
                    value=merged_mot_out,
                    return_attn_weights=True
                )  # cross-attention

                if self.attention_type == LEARNABLEQ_CROSS_SKIP_ATTN:
                    output_tensor = output_tensor + symbolic_input  # skip-connection

            elif self.attention_type == LEARNABLEQ_SELF_CROSS_ATTN:
                # Perform self-attention
                x_self, _ = self._mha2(
                    query=merged_mot_out,
                    key=merged_mot_out,
                    value=merged_mot_out
                )  # self-attention
                merged_mot_out = merged_mot_out + x_self  # skip-connection

                # Expand the learnable query to the desired dimension (seq_len, batch_size, embedding_dim)
                q = self._query
                batch_size_symbolic = symbolic_input.size()[1]
                seq_len_symbolic = symbolic_input.size()[0]
                q = add_to_graph(lambda t, s, b: t.expand(s, b, -1), q, seq_len_symbolic, batch_size_symbolic)

                # Perform cross-attention
                output_tensor, _ = self._mha(
                    query=q,
                    key=merged_mot_out,
                    value=merged_mot_out,
                    return_attn_weights=True
                )  # cross-attention
                output_tensor = output_tensor + symbolic_input  # skip-connection

            else:  # just self-attention
                output_tensor, _ = self._mha(
                    query=merged_mot_out,
                    key=merged_mot_out,
                    value=merged_mot_out
                )  # self-attention
                output_tensor = output_tensor + merged_mot_out  # skip-connection
        else:
            merged_mot_out = Rearrange('s b e -> b s e')(merged_mot_out)
            if self.attention_type == NO_ATTN_SUM:
                output_tensor = merged_mot_out.sum(dim=1)
            elif self.attention_type == NO_ATTN_AVG:
                output_tensor = merged_mot_out.mean(dim=1)
            elif self.attention_type == NO_ATTN_MAX:
                output_tensor = merged_mot_out.max(dim=1)[0]
            elif self.attention_type == NO_ATTN_LEARNABLEQ:
                # Expand the learnable query to the desired dimension (batch_size, embedding_dim, 1)
                q = self._query
                batch_size_symbolic = symbolic_input.size()[1]
                q = add_to_graph(lambda t, b: t.expand(b, -1), q, batch_size_symbolic)
                q = q.unsqueeze(dim=-1)

                # Get the weights (batch_size, num_mot) and normalize them with softmax
                weights = (merged_mot_out @ q).squeeze(-1).softmax(dim=-1)
                output_tensor = (weights.unsqueeze(-1) * merged_mot_out).sum(dim=-1)
                raise NotImplementedError(f"{NO_ATTN_LEARNABLEQ} not implemented yet")
            elif self.attention_type == NO_ATTN_LSTM:
                self._lstm_aggr = torch.nn.LSTM(self.embedding_dim, self.embedding_dim, batch_first=True)
                last_hidden, _ = self._lstm_aggr(merged_mot_out)[1]
                last_hidden = Rearrange('s b e -> b s e')(last_hidden)
                output_tensor = last_hidden[:, -1, ...]
            else:
                raise NotImplementedError(f"Attention type {self.attention_type} not implemented yet")
            output_tensor = output_tensor.unsqueeze(dim=0)

        if verbose:
            print(f"Attention operation allocation completed. Output shape: {output_tensor.shape}. \n"
                  f"Starting final rearrange and normalization operations allocation...")
        output_tensor = Rearrange('s b e -> b s e')(output_tensor)

        if self.input_sequences == INDEPENDENT_INPUT_SEQ and self.attention_type != SELF_ATTN:
            # Reshape from (batch_size*seq_len, 1, embedding_dim) to (batch_size, seq_len, embedding_dim)
            output_tensor = add_to_graph(
                lambda tensor, timesteps: Rearrange('(b s) n e -> b s n e', s=timesteps)(tensor),
                output_tensor,
                n_timesteps
            )
            output_tensor = output_tensor.squeeze(dim=2)

        if self.norm_output:
            output_tensor = torch.nn.LayerNorm(output_tensor.shape[-1])(output_tensor)

        if output_tensor.shape[1] == 1:
            output_tensor = output_tensor.squeeze(1)

        if verbose:
            print(f"Final rearrange and normalization operations allocation. Output shape: {output_tensor.shape}.")

        # Return also sensors, interneurons and motors if required
        if return_all_neurons_output:
            if verbose:
                none_sensors = len(sensors) - non_none_sensors
                none_interneurons = len(interneurons) - non_none_interneurons
                none_motors = len(motors) - non_none_motors
                print(f"Starting allocating the stacking operation on the sensors, interneurons and motors outputs.\n"
                      f"Stacking {non_none_sensors} non-NaN and {none_sensors} NaN tensors (sensors)\n"
                      f"Stacking {non_none_interneurons} non-NaN and {none_interneurons} NaN tensors (interneurons)\n"
                      f"Stacking {non_none_motors} non-NaN and {none_motors} NaN tensors (motors)")

            # Stack sensors, interneurons, motors to a single tensor, replacing None with nan-filled tensors
            sensors = self._replace_none_with_nans(sensors)
            interneurons = self._replace_none_with_nans(interneurons)
            motors = self._replace_none_with_nans(motors)

            merged_sensors = sl.StackLayer(dim=1)(*sensors)
            merged_interneurons = sl.StackLayer(dim=1)(*interneurons)
            merged_motors = sl.StackLayer(dim=1)(*motors)

            if verbose:
                print(f"Stacking operation allocation completed. Output shapes:\n"
                      f"{merged_sensors.shape} (sensors),\n"
                      f"{merged_interneurons.shape} (interneurons),\n"
                      f"{merged_motors.shape} (motors)")

            return output_tensor, merged_sensors, merged_interneurons, merged_motors

        return output_tensor
