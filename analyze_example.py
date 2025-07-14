import os
import pickle
from tqdm import tqdm
import io

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

import train
import preprocessing
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization


"""
This file allows you to train one model on one task, and see plots of what
the process and end result looks like. You can input the training split and
the task code, and it will:
- Train a model for the specified number of steps,
- Create a GIF showing sampled solutions from the model at every 100 steps,
  with each frame displayed for 2 seconds,
- Plot the KL and reconstruction error over time,
- Plot the contribution of each tensor shape to the KL over time,
- Show top principal components of each tensor that still contributes to
  the KL at the end of training.
"""

# For some reason trying to set the seed doesn't actually fix results.
# Just run things over and over again until you see desired interesting behaviors.
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

if __name__ == "__main__":

    # Some interesting tasks: 272f95fa, 6d75e8bb, 6cdd2623, 41e4d17e, 2bee17df
    # 228f6490, 508bd3b6, 2281f1f4, ecdecbb3
    split = input('Enter which split you want to find the task in (training (def), evaluation, test): ')
    if split.strip() == '':
        split = 'training'
    task_name = input('Enter which task you want to analyze (def: 272f95fa): ')
    if task_name.strip() == '':
        task_name = '272f95fa'
    folder = f'results/{task_name}/'

    save_dir = f'results/{task_name}'

    n_iterations = input('Enter number of iterations (def: 500): ')
    if n_iterations.strip() == '':
        n_iterations = 500
    else:
        n_iterations = int(n_iterations)
    print('Performing a training run on task', task_name,
          'and placing the results in', folder)
    os.makedirs(folder, exist_ok=True)

    # Preprocess the task, set up the training
    task = preprocessing.preprocess_tasks(split, [task_name])[0]
    model = arc_compressor.ARCCompressor(task)
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    train_history_logger = solution_selection.Logger(task)
    visualization.plot_problem(train_history_logger)
    
    # Store frames for GIF creation
    gif_frames = []
    plot_every_steps = 100
    
    for train_step in tqdm(range(n_iterations)):
        train.take_step(task, model, optimizer, train_step, train_history_logger)
        
        if (train_step+1) % plot_every_steps == 0:
            # Create a temporary filename for this frame
            temp_fname = f'{save_dir}/temp_frame.png'
            
            # Generate the plot and save temporarily
            visualization.plot_solution(train_history_logger, fname=temp_fname)
            
            # Read the image and add to gif frames
            frame = imageio.imread(temp_fname)
            gif_frames.append(frame)
            
            # Clean up temporary file
            os.remove(temp_fname)
    
    if gif_frames:
        gif_filename = f'{save_dir}/training_progress.gif'
        frame_dur_s = 1.0
        imageio.mimsave(gif_filename, gif_frames, fps=1/frame_dur_s)
        print(f'Training progress GIF saved as: {gif_filename}')

    # Save the metrics, model weights, and learned representations.
    np.savez(f'{save_dir}/KL_curves.npz',
             KL_curves={key:np.array(val) for key, val in train_history_logger.KL_curves.items()},
             reconstruction_error_curve=np.array(train_history_logger.reconstruction_error_curve),
             multiposteriors=model.multiposteriors,
             target_capacities=model.target_capacities,
             decode_weights=model.decode_weights)

    # Load the metrics, model weights, and learned representations.
    stored_data = np.load(f'{save_dir}/KL_curves.npz', allow_pickle=True)
    KL_curves = stored_data['KL_curves'][()]
    reconstruction_error_curve = stored_data['reconstruction_error_curve']
    multiposteriors = stored_data['multiposteriors'][()]
    target_capacities = stored_data['target_capacities'][()]
    decode_weights = stored_data['decode_weights'][()]
    
    # Plot the KL curves over time.
    # Sort curves by their final values and assign colors/labels to top 5
    
    # Get final values for each curve and sort
    curve_final_values = []
    for component_name, curve in KL_curves.items():
        final_value = curve[-1] if len(curve) > 0 else 0
        curve_final_values.append((component_name, curve, final_value))
    
    # Sort by final value (descending)
    curve_final_values.sort(key=lambda x: x[2], reverse=True)
    
    # Define colors for top 5 curves (easy to distinguish)
    top_colors = [
        (1, 0, 0),      # Red
        (0, 0, 1),      # Blue  
        (0, 0.8, 0),    # Green
        (1, 0.5, 0),    # Orange
        (0.8, 0, 0.8),  # Purple
    ]
    
    fig, ax = plt.subplots()
    has_labels = False
    
    for i, (component_name, curve, final_value) in enumerate(curve_final_values):
        if i < 5:  # Top 5 curves get colors and labels
            line_color = top_colors[i]
            # Create label from component dimensions
            dims = tuple(eval(component_name))
            axis_names = ['example', 'color', 'direction', 'height', 'width']
            axis_names = [axis_name
                for axis_name, axis_exists in zip(axis_names, dims) if axis_exists]
            label = '(' + ', '.join(axis_names) + ', channel)'
            has_labels = True
        else:  # Rest get gray color, no label
            line_color = (0.5, 0.5, 0.5)
            label = None
        
        ax.plot(np.arange(curve.shape[0]), curve, color=line_color, label=label)

    if has_labels:
        ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('KL contribution')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(f'{save_dir}/KL_components.png', bbox_inches='tight')
    plt.close()

    # Plot the KL vs reconstruction error
    fig, ax = plt.subplots()
    total_KL = 0
    for component_name, curve in KL_curves.items():
        total_KL = total_KL + curve
    fig, ax = plt.subplots()
    ax.plot(np.arange(total_KL.shape[0]), total_KL, label='KL from z', color='k')
    ax.plot(np.arange(reconstruction_error_curve.shape[0]),
            reconstruction_error_curve, label='reconstruction error', color='r')
    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('total KL or reconstruction error')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(f'{save_dir}/KL_vs_reconstruction.png', bbox_inches='tight')
    plt.close()

    # Get the learned representation tensors
    samples = []
    for i in range(100):
        sample, KL_amounts, KL_names = layers.decode_latents(target_capacities,
                                           decode_weights, multiposteriors)
        samples.append(sample)

    def average_samples(dims, *items):
        mean = torch.mean(torch.stack(items, dim=0), dim=0).detach().cpu().numpy()
        all_but_last_dim = tuple(range(len(mean.shape) - 1))
        mean = mean - np.mean(mean, axis=all_but_last_dim)
        return mean
    means = multitensor_systems.multify(average_samples)(*samples)

    # Figure out which tensors contain significant information
    dims_to_plot = []
    for KL_amount, KL_name in zip(KL_amounts, KL_names):
        dims = tuple(eval(KL_name))
        if torch.sum(KL_amount).detach().cpu().numpy() > 1:
            dims_to_plot.append(dims)

    # Show the top principal components of the significant tensors.
    color_names = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'light blue', 'brown']
    restricted_color_names = [color_names[i] for i in task.colors]
    restricted_color_codes = [tuple((visualization.color_list[i]/255).tolist())
                              for i in task.colors]
    for dims in dims_to_plot:
        tensor = means[dims]

        max_components = 1
        cutoff = 0.01
        direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]

        orig_shape = tensor.shape
        if len(orig_shape) == 2:
            tensor = tensor[None,:,:]
        orig_shape = tensor.shape
        if len(orig_shape) == 3:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get top 3 principal components
            for component_num in range(max_components):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]  # Calculate component strength
                if strength < cutoff:
                    continue

                # Show the component
                fig, ax = plt.subplots()
                ax.imshow(component, cmap='gray', vmin=-1, vmax=1)
                
                # Pick the axis labels
                axis_names = ['example', 'color', 'direction', 'height', 'width']
                tensor_name = '_'.join([axis_name
                    for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                if sum(dims) == 2:
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                else:
                    x_dim = None
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                plt.ylabel(x_dim)
                plt.xlabel(y_dim)

                if x_dim is None:
                    ax.set_yticks([])
                    ax.set_xticks([], minor=True)
                if y_dim is None:
                    ax.set_xticks([])
                    ax.set_xticks([], minor=True)

                # Set the tick labels
                # Tick labels for example axis
                if x_dim == 'example':
                    ax.set_yticks(np.arange(task.n_examples))
                if y_dim == 'example':
                    ax.set_xticks(np.arange(task.n_examples))

                # Tick labels for color axis
                if x_dim == 'color':
                    ax.set_yticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_yticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_yticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")
                if y_dim == 'color':
                    ax.set_xticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_xticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_xticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")

                # Tick labels for direction axis
                if x_dim == 'direction':
                    ax.set_yticks(np.arange(8))
                    ax.set_yticklabels(direction_names)
                    ax.tick_params(axis='y', which='major', labelsize=22)
                if y_dim == 'direction':
                    ax.set_xticks(np.arange(8))
                    ax.set_xticklabels(direction_names)
                    ax.tick_params(axis='x', which='major', labelsize=22)

                # Standard tick labels for height and width axes

                ax.set_title('component' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.savefig(f'{save_dir}/{tensor_name}_component_{component_num}.png', bbox_inches='tight')
                plt.close()

        # Plot an ({example, color, direction}, x, y) tensor with subplots
        elif len(orig_shape) == 4 and dims[3] == 1 and dims[4] == 1:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get the top 3 principal components
            for component_num in range(max_components):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]
                if strength < cutoff:
                    continue
                n_plots = orig_shape[0]

                # Make the subplots
                fig, axs = plt.subplots(1, n_plots)
                for plot_idx in range(n_plots):
                    ax = axs[plot_idx]
                    ax.imshow(component[plot_idx,:,:], cmap='gray', vmin=-1, vmax=1)

                    # Get the axis labels
                    axis_names = ['example', 'color', 'direction', 'height', 'width']
                    tensor_name = '_'.join([axis_name
                        for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                    ax_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][2]
                    ax.set_ylabel(x_dim)
                    ax.set_xlabel(y_dim)

                    # Standard tick labels for height and width axes

                    # Label the subplots
                    if ax_dim == 'example':
                        ax.set_title('example ' + str(plot_idx))
                    elif ax_dim == 'color':
                        ax.set_title(restricted_color_names[plot_idx],
                                     color=restricted_color_codes[plot_idx],
                                     fontweight="bold")
                    elif ax_dim == 'direction':
                        ax.set_title(direction_names[plot_idx], fontsize=22)

                plt.subplots_adjust(wspace=1)
                fig.suptitle('component ' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.subplots_adjust(top=1.4)
                plt.savefig(f'{save_dir}/{tensor_name}_component_{component_num}.png', bbox_inches='tight')
                plt.close()


print('done :)')
