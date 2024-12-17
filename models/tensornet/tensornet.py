from typing import Optional, Union, Dict, List, Tuple, Final, Callable, Iterable, FrozenSet
import igraph
import torch
import einops
from models.layers.misc import SerializableModule
from models.tensornet.tensor_units import LinearNodeTensorUnit, SENSOR, INTERNEURON, MOTOR
from models.layers.attention_utils import find_n_heads, find_mha_embedding_dim, CustomMultiHeadAttention
from utils.connectome_reader import ConnectomeReader


N_HEADS: Final[int] = 8
HEAD_DIM: Final[int] = 32
ADD_SKIP_CONN: Final[str] = "add"
MULT_SKIP_CONN: Final[str] = "mult"
SENSOR_SENSOR: Final[str] = "SS"
SENSOR_INTERNEURON: Final[str] = "SI"
SENSOR_MOTOR: Final[str] = "SM"
INTERNEURON_SENSOR: Final[str] = "IS"
INTERNEURON_INTERNEURON: Final[str] = "II"
INTERNEURON_MOTOR: Final[str] = "IM"
MOTOR_SENSOR: Final[str] = "MS"
MOTOR_INTERNEURON: Final[str] = "MI"
MOTOR_MOTOR: Final[str] = "MM"
PROCESSING_ORDER_DEFAULT: Final[str] = "S->O,I->O,M->O,S->S,I->I,M->M"
CROSS_ATTN: Final[str] = "cross"
CROSS_ATTN_SKIP_CONN: Final[str] = "cross-skip"
SELF_ATTN: Final[str] = "self"
SELF_CROSS_ATTN: Final[str] = "self-cross"


class TensorNetwork(SerializableModule):
    ACTIVATIONS: Final[Dict[str, Callable]] = {
        "linear": torch.nn.Identity(),
        "relu": torch.nn.ReLU(),
        "leaky_relu": torch.nn.LeakyReLU(),
        "rrelu": torch.nn.RReLU(),
        "relu6": torch.nn.ReLU6(),
        "gelu": torch.nn.GELU(),
        "elu": torch.nn.ELU(),
        "celu": torch.nn.CELU(),
        "glu": torch.nn.GLU(),
        "selu": torch.nn.SELU(),
        "prelu": torch.nn.PReLU(),
        "silu": torch.nn.SiLU(),
        "hardswish": torch.nn.Hardswish(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "log_sigmoid": torch.nn.LogSigmoid(),
        "softmax": torch.nn.Softmax(dim=-1),
        "hardtanh": torch.nn.Hardtanh()
    }

    def __init__(self,
                 source: Union[str, ConnectomeReader],
                 input_dim: int,
                 embedding_dim: int,
                 activation: str = "gelu",
                 n_heads: Optional[int] = None,
                 head_dim: Optional[int] = HEAD_DIM,
                 dropout_mha: float = 0.0,
                 attention_type: str = "cross",
                 adjust_n_heads: bool = True,
                 norm_output: bool = True,
                 norm_tensor_unit: bool = False,
                 skip_connection_type: str = MULT_SKIP_CONN,
                 reapply_unit_synapses: Optional[Iterable[str]] = None,
                 skip_conn_only_synapses: Optional[Iterable[str]] = None,
                 ignored_synapses: Optional[Iterable[str]] = None,
                 processing_order: Optional[str] = PROCESSING_ORDER_DEFAULT,
                 device: Optional[Union[str, torch.device]] = None):
        """
        The TensorNetwork module designed to emulate the structure and topology of natural connectomes. The model
        consists of interconnected sensor, interneuron, and motor neurons, with heterophilic (between different neuron
        types) and homophilic (between the same neuron types) connections.

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
        :param adjust_n_heads: a boolean indicating whether to adjust the number of heads or the embedding dimension to
            make it divisible by the number of heads in multi-head attention layer. Only used if head_dim is not given.
            Default is False.
        :type adjust_n_heads: bool
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
            processed, separated by comma ("S->O" stands for sensor to others, "S->S" for sensor to sensor, "I->O" for
            interneuron to others, "I->I" for interneuron to interneuron, "M->O" for motor to others and "M->M" for
            motor to motor). Default is "S->O,I->O,M->O,S->S,I->I,M->M").
        :type processing_order: str
        :param device: The `device` parameter is an optional argument that specifies the device on which the
            computations will be performed. It can be a string specifying the device (e.g. "cpu" or "cuda"), or a
            torch.device object. If not specified, the default device will be used.
        """
        super().__init__()

        # Read the connectome file
        if isinstance(source, str):
            source = ConnectomeReader(name_file=source)
        source.read()

        # Init instance variables
        self.__connectome_filename: str = source.name_file
        self.__input_dim: int = input_dim
        self.__embedding_dim: int = embedding_dim
        self.__adjusted_embedding_dim: int = embedding_dim
        self.__head_dim: Optional[int] = head_dim
        self.__activation: str = activation
        self.__n_heads: Optional[int] = n_heads
        self.__dropout_mha: float = dropout_mha
        self.__attention_type: str = attention_type
        self.__adjust_n_heads: bool = adjust_n_heads
        self.__norm_output: bool = norm_output
        self.__norm_tensor_unit: bool = norm_tensor_unit
        self.__connectome_graph: igraph.Graph = source.graph_el
        self.__neuron_type_map: dict = source.get_node_role_dict()
        self.__skip_connection_type: str = skip_connection_type
        self.__reapply_unit_synapses: FrozenSet[str] = frozenset(
            reapply_unit_synapses if reapply_unit_synapses else []
        )
        self.__skip_conn_only_synapses: FrozenSet[str] = frozenset(
            skip_conn_only_synapses if skip_conn_only_synapses else []
        )
        self.__ignored_synapses: FrozenSet[str] = frozenset(
            ignored_synapses if ignored_synapses else []
        )
        self.__processing_order: str = processing_order
        self._ignored_connections_count: int = 0  # ignored connection count of the last forward pass, here for debug

        # Count number of motor neurons for MHA heads
        self.__n_motors: int = 0
        for u in self.__neuron_type_map:
            if self.__neuron_type_map[u] == MOTOR:
                self.__n_motors += 1

        # Multi-head attention with n_heads heads if given, otherwise just use the number of motors
        if n_heads is None:
            n_heads = self.__n_motors

        # If head dimension is given, make an additional projection over queries to adjust MHA dimension
        if head_dim is not None:
            # Effective embedding size will be n_head*head_dim
            self._mha = CustomMultiHeadAttention(
                embed_dim=embedding_dim,
                head_dim_key=head_dim,
                n_heads=n_heads,
                dropout=dropout_mha,
                batch_first=False
            )
        else:
            # Adjust the embedding dimension or the number of heads if the former is not divisible by the latter
            if embedding_dim % n_heads != 0:
                if adjust_n_heads:
                    n_heads = find_n_heads(embed_dim=embedding_dim, max_n_heads=n_heads)
                else:
                    embedding_dim = find_mha_embedding_dim(embed_dim=embedding_dim, n_heads=n_heads)
                    self.__adjusted_embedding_dim = embedding_dim  # update the instance variable

            # Create MHA layer
            self._mha = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=n_heads,
                dropout=dropout_mha,
                device=device
            )

        # Add a projection to the embedding dimension if needed
        if input_dim != embedding_dim:
            self._projection = torch.nn.Linear(in_features=input_dim, out_features=embedding_dim)
        else:
            self._projection = None

        # Add layer normalization
        self._layer_norm = torch.nn.LayerNorm(embedding_dim)

        # Init the module containers for the different types of neurons
        self._sensor_neurons = torch.nn.ModuleDict()
        self._interneurons = torch.nn.ModuleDict()
        self._motor_neurons = torch.nn.ModuleDict()
        self._activation_fn = self.ACTIVATIONS[activation]

        # Init all the layers representing the neurons
        for neuron in self.__neuron_type_map:
            neuron_type = self.__neuron_type_map[neuron]
            if neuron_type == SENSOR:
                module_dict = self._sensor_neurons
            elif neuron_type == INTERNEURON:
                module_dict = self._interneurons
            else:
                module_dict = self._motor_neurons

            tensor_unit = LinearNodeTensorUnit(in_features=embedding_dim,
                                               out_features=embedding_dim,
                                               neuron_type=neuron_type)
            if norm_tensor_unit:
                module_dict[str(neuron)] = torch.nn.Sequential(
                    tensor_unit,
                    torch.nn.LayerNorm(embedding_dim)
                )
            else:
                module_dict[str(neuron)] = tensor_unit

        # Store the various kinds of heterophilic/homophilic connections, scanning the adjacency matrix
        self.__sensor_to_others = []
        self.__interneuron_to_others = []
        self.__motor_to_others = []
        self.__sensor_to_sensor = []
        self.__interneuron_to_interneuron = []
        self.__motor_to_motor = []

        # This is O(|V|^2), maybe it could be reduced to O(|E|) being given the edgelist instead of the adj
        adj = self.adj
        for u in range(0, self.connectome_graph.vcount()):
            for v in range(0, self.connectome_graph.vcount()):
                u_neuron_type = self.__neuron_type_map[u]
                v_neuron_type = self.__neuron_type_map[v]

                if adj[u, v] != 0:
                    if u_neuron_type == SENSOR and v_neuron_type == SENSOR:
                        if SENSOR_SENSOR not in self.ignored_synapses:
                            self.__sensor_to_sensor.append((str(u), str(v)))

                    elif u_neuron_type == SENSOR and v_neuron_type != SENSOR:
                        if (v_neuron_type == INTERNEURON and SENSOR_INTERNEURON not in self.ignored_synapses) or \
                                (v_neuron_type == MOTOR and SENSOR_MOTOR not in self.ignored_synapses):
                            self.__sensor_to_others.append((str(u), str(v)))

                    elif u_neuron_type == INTERNEURON and v_neuron_type == INTERNEURON:
                        if INTERNEURON_INTERNEURON not in self.ignored_synapses:
                            self.__interneuron_to_interneuron.append((str(u), str(v)))

                    elif u_neuron_type == INTERNEURON and v_neuron_type != INTERNEURON:
                        if (v_neuron_type == SENSOR and INTERNEURON_SENSOR not in self.ignored_synapses) or \
                                (v_neuron_type == MOTOR and INTERNEURON_MOTOR not in self.ignored_synapses):
                            self.__interneuron_to_others.append((str(u), str(v)))

                    elif u_neuron_type == MOTOR and v_neuron_type == MOTOR:
                        self.__motor_to_motor.append((str(u), str(v)))
                    else:
                        if (v_neuron_type == SENSOR and MOTOR_SENSOR not in self.ignored_synapses) or \
                                (v_neuron_type == MOTOR and INTERNEURON_MOTOR not in self.ignored_synapses):
                            self.__motor_to_others.append((str(u), str(v)))

    @property
    def adj(self) -> torch.Tensor:
        return torch.tensor(self.__connectome_graph.get_adjacency().data)

    @property
    def connectome_graph(self) -> igraph.Graph:
        return self.__connectome_graph

    @property
    def neuron_type_map(self) -> dict[int, str]:
        return self.__neuron_type_map

    @property
    def embedding_dim(self) -> int:
        return self.__adjusted_embedding_dim

    @property
    def input_dim(self) -> int:
        return self.__input_dim

    @property
    def activation(self) -> str:
        return self.__activation

    @property
    def n_motors(self) -> int:
        return self.__n_motors

    @property
    def n_heads(self) -> int:
        return self._mha.num_heads

    @property
    def head_dim(self) -> int:
        return self._mha.head_dim

    @property
    def dropout_mha(self) -> float:
        return self.__dropout_mha

    @property
    def attention_type(self) -> str:
        return self.__attention_type

    @property
    def adjust_n_heads(self) -> bool:
        return self.__adjust_n_heads

    @property
    def connectome_filename(self) -> str:
        return self.__connectome_filename

    @property
    def norm_output(self) -> bool:
        return self.__norm_output

    @property
    def norm_tensor_unit(self) -> bool:
        return self.__norm_tensor_unit

    @property
    def skip_connection_type(self) -> str:
        return self.__skip_connection_type

    @property
    def reapply_unit_synapses(self) -> FrozenSet[str]:
        return self.__reapply_unit_synapses

    @property
    def skip_conn_only_synapses(self) -> FrozenSet[str]:
        return self.__skip_conn_only_synapses

    @property
    def ignored_synapses(self) -> FrozenSet[str]:
        return self.__ignored_synapses

    @property
    def processing_order(self) -> str:
        return self.__processing_order

    @property
    def sensor_to_others(self) -> List[Tuple[str, str]]:
        return self.__sensor_to_others

    @property
    def sensor_to_sensor(self) -> List[Tuple[str, str]]:
        return self.__sensor_to_sensor

    @property
    def interneuron_to_others(self) -> List[Tuple[str, str]]:
        return self.__interneuron_to_others

    @property
    def interneuron_to_interneuron(self) -> List[Tuple[str, str]]:
        return self.__interneuron_to_interneuron

    @property
    def motor_to_others(self) -> List[Tuple[str, str]]:
        return self.__motor_to_others

    @property
    def motor_to_motor(self) -> List[Tuple[str, str]]:
        return self.__motor_to_motor

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "source": self.connectome_filename,
            "input_dim": self.input_dim,
            "embedding_dim": self.__embedding_dim,  # gets the one given to the constructor if it has been adjusted
            "head_dim": self.__head_dim,  # gets the one given to the constructor
            "activation": self.activation,
            "n_heads": self.__n_heads,  # gets the one given to the constructor if it has been adjusted
            "dropout_mha": self.dropout_mha,
            "attention_type": self.attention_type,
            "adjust_n_heads": self.adjust_n_heads,
            "norm_output": self.norm_output,
            "norm_tensor_unit": self.norm_tensor_unit,
            "skip_connection_type": self.skip_connection_type,
            "reapply_unit_synapses": self.reapply_unit_synapses,
            "skip_conn_only_synapses": self.skip_conn_only_synapses,
            "ignored_synapses": self.ignored_synapses,
            "processing_order": self.processing_order
        }

    @classmethod
    def from_constructor_params(cls, constructor_params: dict, *args, **kwargs):
        """
        Takes a dictionary of constructor parameters and returns an instance of a class with those
        parameters, along with an optional device parameter.

        :param constructor_params: A dictionary containing the parameters that will be passed to the constructor of the
            class.
        :type constructor_params: dict

        :return: The method is returning an instance of the class `cls` with the `device` parameter set to the value of
            the `device` kwarg, if it exists, and with the remaining constructor parameters passed in as
            `constructor_params`.
        """
        device = None
        if "device" in constructor_params:
            device = constructor_params["device"]
            del constructor_params["device"]
        if "device" in kwargs:
            device = kwargs["device"]
            del kwargs["device"]

        return cls(*args, device=device, **constructor_params, **kwargs)

    def _get_processing_ordered_connections(self) -> List[List[Tuple[int, int]]]:

        # Get connections in dictionary
        all_connections_dict = {
            "S->O": self.__sensor_to_others,
            "I->O": self.__interneuron_to_others,
            "M->O": self.__motor_to_others,
            "S->S": self.__sensor_to_sensor,
            "I->I": self.__interneuron_to_interneuron,
            "M->M": self.__motor_to_motor
        }

        # Order connections according to the given processing order
        ordered_connection_types = self.processing_order.split(",")
        processing_ordered_connections: List[List[Tuple[int, int]]] = [
            all_connections_dict[conn_type] for conn_type in ordered_connection_types
        ]

        return processing_ordered_connections

    def get_connection_type(self, u: int, v: int) -> Optional[str]:

        adj = self.adj
        if adj[u, v] != 0:
            u_type = self.neuron_type_map[u]
            v_type = self.neuron_type_map[v]

            if u_type == SENSOR and v_type == SENSOR:
                return SENSOR_SENSOR
            elif u_type == SENSOR and v_type == INTERNEURON:
                return SENSOR_INTERNEURON
            elif u_type == SENSOR and v_type == MOTOR:
                return SENSOR_MOTOR
            elif u_type == INTERNEURON and v_type == SENSOR:
                return INTERNEURON_SENSOR
            elif u_type == INTERNEURON and v_type == INTERNEURON:
                return INTERNEURON_INTERNEURON
            elif u_type == INTERNEURON and v_type == MOTOR:
                return INTERNEURON_MOTOR
            elif u_type == MOTOR and v_type == SENSOR:
                return MOTOR_SENSOR
            elif u_type == MOTOR and v_type == INTERNEURON:
                return MOTOR_INTERNEURON
            elif u_type == MOTOR and v_type == MOTOR:
                return MOTOR_MOTOR
        return None

    # TODO: test this (gotta be pretty complex)
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
            Performs the forward pass of the tensor network model.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_size) or (batch_size, seq_len, input_size)
                    representing the sensory input.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, motor_size) representing the motor output.

            Raises:
                AssertionError: If the dimensions of the input tensor `x` do not match the expected input size.

            Notes:
                This method implements the forward pass of the tensor network model, designed to emulate the structure
                and topology of natural connectomes.

                During it, the input tensor `x` is first passed through an optional projection layer, if defined, to
                match the desired input size. Then, the model propagates the input through the graph of neurons,
                applying dense layers and element-wise multiplication or addition based on the connection types.
                The sensory neurons receive the initial input, while the interneurons and motor neurons receive the
                outputs from other neurons.

                The model traverses the heterophilic and homophilic connections in a specific order: first from sensors
                to other neurons, then from interneurons to other neurons, and finally from motors to other neurons.
                This ensures that the outputs from preceding neurons are available as inputs for the subsequent neurons.

                If a neuron's output is None or is a sensory neuron's initial input, a target neuron's dense layer is
                applied to compute its output. If a neuron's output is already computed, the model performs element-wise
                multiplication or addition between the neuron's input and output tensors.

                Finally, the motor outputs are aggregated using multi-head attention, which captures the relationships
                between different motor neurons and produces a consolidated motor output tensor. This is done by either
                performing self-attention on the outputs of the motor neurons, or a cross-attention operation between
                the inputs and the motor outputs. Additionally, from a biological perspective, it has been shown by
                Whittington et al. (https://arxiv.org/abs/2112.04035) that the multi-head attention mechanism produces
                similar neural activation to the brain hyppocampus, which biologically aggregates the information coming
                from different neural sources.

                The expected dimensions of the input tensor `x` and the sizes of sensor, interneuron, and motor neurons
                should be properly set before calling this method. The graph structure and connection weights should
                also be initialized.

            Examples:
                >>> model = TensorNetwork("celegans.graphml", 256, 128, n_heads=32, dropout_mha=0.3)
                >>> input_tensor = torch.randn(64, 256)  # Batch size of 64, input size of 256
                >>> output_tensor = model(input_tensor)
            """

        # Apply projection if required
        if self._projection is not None:
            x = self._projection(x)
        stored_input = x

        # Initialize empty node output (input for sensor nodes)
        # Tuple indicating that the node content is just initialized to the input
        sensor_output_tensors = {str(u): (x,) for u in self._sensor_neurons}
        interneuron_output_tensors = {str(u): None for u in self._interneurons}
        motor_output_tensors = {str(u): None for u in self._motor_neurons}

        # Traverse the heterophilic/homophilic connections, according to the given connection ordering
        all_connections = self._get_processing_ordered_connections()

        # This should be O(|E|), gotta test it tough to see if it could be made easier to optimize for torch.compile()
        self._ignored_connections_count = 0
        for connections in all_connections:

            # Check if connections should be ignored
            for u, v in connections:

                # Get the input/output tensors
                if u in sensor_output_tensors:
                    node_input_tensor = sensor_output_tensors[u]
                elif u in interneuron_output_tensors:
                    node_input_tensor = interneuron_output_tensors[u]
                else:
                    node_input_tensor = motor_output_tensors[u]
                if v in sensor_output_tensors:
                    node_output_tensor = sensor_output_tensors[v]
                    output_dense_layer = self._sensor_neurons[str(v)]
                    output_tensors_dict = sensor_output_tensors
                elif v in interneuron_output_tensors:
                    node_output_tensor = interneuron_output_tensors[v]
                    output_dense_layer = self._interneurons[str(v)]
                    output_tensors_dict = interneuron_output_tensors
                else:
                    node_output_tensor = motor_output_tensors[v]
                    output_dense_layer = self._motor_neurons[str(v)]
                    output_tensors_dict = motor_output_tensors

                connection_type = self.get_connection_type(int(u), int(v))
                if node_input_tensor is not None:
                    # If the node output is None or is a sensor initial input, then apply the target node dense layer
                    if node_output_tensor is None or isinstance(node_output_tensor, tuple):

                        # If the input is a tuple, then extract the tensor from it before applying the dense layer
                        if isinstance(node_input_tensor, tuple):
                            node_input_tensor = node_input_tensor[0]

                        # Apply dense layer (+ layer normalization if required) & activation
                        output_tensors_dict[v] = self._activation_fn(output_dense_layer(node_input_tensor))

                    # Otherwise, employ the skip-connection
                    else:
                        if isinstance(node_input_tensor, tuple):
                            node_input_tensor = node_input_tensor[0]

                        # For the linear-only connections apply the linear layer again with additive skip-connection
                        if connection_type in self.reapply_unit_synapses:
                            # Get the output node's linear layer
                            dense_layer = output_dense_layer
                            if self.norm_tensor_unit:
                                # Don't get layer normalization, will normalize later
                                dense_layer = output_dense_layer[0]

                            # Apply the dense layer
                            projected_node_input_tensor = self._activation_fn(dense_layer(node_input_tensor))

                            # Perform additive skip-connection
                            output_tensors_dict[v] = torch.add(projected_node_input_tensor, node_output_tensor)

                        # Otherwise, apply additive/multiplicative skip-connection
                        elif self.skip_connection_type == ADD_SKIP_CONN:
                            output_tensors_dict[v] = torch.add(node_input_tensor, node_output_tensor)
                        else:
                            output_tensors_dict[v] = torch.mul(node_input_tensor, node_output_tensor)

                        if self.norm_tensor_unit:
                            layer_norm = output_dense_layer[1]  # get from the torch.nn.Sequential()
                            output_tensors_dict[v] = layer_norm(output_tensors_dict[v])
                else:
                    self._ignored_connections_count += 1
                    # TODO: maybe add else to handle "ignored" connection when input tensor is None

        # Get the output vector from each motor, stacking them all on the 2th dimension
        motor_output_tensors = [
            motor_output_tensors[u] for u in motor_output_tensors if motor_output_tensors[u] is not None
        ]
        x = torch.stack(motor_output_tensors, dim=1)

        # Apply multi-head attention to aggregate the motor outputs
        if len(x.shape) == 2:
            # If output was not a sequence, having shape (B, E), unsqueeze it to shape (B, 1, E)
            x = x.unsqueeze(1)
        x = einops.rearrange(x, "b s e -> s b e")

        if self.attention_type == CROSS_ATTN or self.attention_type == CROSS_ATTN_SKIP_CONN:
            if len(stored_input.shape) == 2:
                # If input was not a sequence, having shape (B, E), unsqueeze it to shape (B, 1, E)
                stored_input = stored_input.unsqueeze(1)
            stored_input = einops.rearrange(stored_input, "b s e -> s b e")
            x, _ = self._mha(query=stored_input, key=x, value=x)  # cross-attention

            if self.attention_type == CROSS_ATTN_SKIP_CONN:
                x = x + stored_input

        elif self.attention_type == SELF_CROSS_ATTN:

            # Perform self-attention
            x_self, _ = self._mha(query=x, key=x, value=x)  # self-attention
            x = x + x_self  # skip-connection

            # Perform cross-attention
            if len(stored_input.shape) == 2:
                # If input was not a sequence, having shape (B, E), unsqueeze it to shape (B, 1, E)
                stored_input = stored_input.unsqueeze(1)
            stored_input = einops.rearrange(stored_input, "b s e -> s b e")
            x, _ = self._mha(query=stored_input, key=x, value=x)  # cross-attention
            x = x + stored_input  # skip-connection
        else:
            x_self, _ = self._mha(query=x, key=x, value=x)  # self-attention
            x = x + x_self  # skip-connection

        x = einops.rearrange(x, "s b e -> b s e")  # gotta test if we need this
        x = x.squeeze(1)  # (B, 1, E) -> (B, E)

        # Normalize output if required
        if self.norm_output:
            x = self._layer_norm(x)

        return x
