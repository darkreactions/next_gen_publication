import plotly.graph_objs as go
import pandas as pd
import os
from ipywidgets import Tab, SelectMultiple, Accordion, ToggleButton, VBox, HBox, HTML, Image, Button, Text, Dropdown
from ipywidgets import HBox, VBox, Image, Layout, HTML
import numpy as np
import ast
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn import preprocessing
import math
import json


class Figure1:
    def __init__(self, csv_file_path, base_path='', inchi_key='XFYICZOIWSBQSK-UHFFFAOYSA-N', cluster=None):
        self.selected_plate = None
        self.old_plate = None
        self.base_path = base_path
        self.clustering = cluster
        self.full_perovskite_data = pd.read_csv(
            csv_file_path, low_memory=False, skiprows=4)
        self.full_perovskite_data = self.full_perovskite_data[
            self.full_perovskite_data['_raw_ExpVer'] >= 1.1]
        self.organic_inchis = pd.read_csv(
            './perovskitedata/organic_inchikey.csv', sep='\t')
        self.solvent_inchis = json.load(open('./perovskitedata/solvents.json'))

        # Filtering inchis that exist in full_perovskite_data

        self.state_spaces = pd.read_csv('./perovskitedata/state_spaces.csv')
        self.ss_dict = {}
        for row in self.state_spaces.iterrows():
            row = list(row[1])
            # app.logger.info(row)
            points = [ast.literal_eval(pt) for pt in row[1:-1]]
            chemical_abbrev = ast.literal_eval(row[-1])[1]
            inchi_row = self.organic_inchis[self.organic_inchis['Chemical Abbreviation']
                                            == chemical_abbrev]
            if not inchi_row.empty:
                row = inchi_row.iloc[0]
                self.ss_dict[row['InChI Key (ID)']] = points
        self.generate_plot(inchi_key, '')
        self.setup_widgets()

    def generate_plot(self, inchi_key, solvent_inchi):
        if inchi_key in self.ss_dict:
            self.setup_hull(hull_points=self.ss_dict[inchi_key])
        else:
            self.setup_hull(hull_points=[[0, 0, 0]])
        self.gen_amine_traces(inchi_key, solvent=solvent_inchi)
        self.setup_plot()

    def setup_hull(self, hull_points=[[0., 0., 0.], [0., 2.3, 0.], [1.86, 1.86, 0.],
                                      [0., 0., 9.5], [1.19339, 1.19339, 9.5], [0., 1.4757, 9.5]]):
        xp, yp, zp = zip(*hull_points)
        self.hull_mesh = go.Mesh3d(x=xp,
                                   y=yp,
                                   z=zp,
                                   color='green',
                                   opacity=0.50,
                                   alphahull=0)

    def gen_amine_traces(self, inchi_key, amine_short_name='Me2NH2I', solvent=None):
        amine_data = self.full_perovskite_data.loc[self.full_perovskite_data['_rxn_organic-inchikey'] == inchi_key]
        if solvent:
            amine_data = amine_data[amine_data['_raw_reagent_0_chemicals_0_InChIKey'] == solvent]
        print(f'Total points: {len(amine_data)}')
        self.max_inorg = amine_data['_rxn_M_inorganic'].max()
        self.max_org = amine_data['_rxn_M_organic'].max()
        self.max_acid = amine_data['_rxn_M_acid'].max()

        # Splitting by crystal scores. Assuming crystal scores from 1-4
        self.amine_crystal_dfs = []
        for i in range(1, 5):
            self.amine_crystal_dfs.append(
                amine_data.loc[amine_data['_out_crystalscore'] == i])

        self.amine_crystal_traces = []
        self.trace_colors = ['rgba(65, 118, 244, 1.0)', 'rgba(92, 244, 65, 1.0)',
                             'rgba(244, 238, 66, 1.0)', 'rgba(244, 66, 66, 1.0)']
        for i, df in enumerate(self.amine_crystal_dfs):
            trace = go.Scatter3d(
                x=df['_rxn_M_inorganic'],
                y=df['_rxn_M_organic'],
                z=df['_rxn_M_acid'],
                mode='markers',
                name='Score {}'.format(i+1),
                text=['<b>PbI3</b>: {:.3f} <br><b>Amine</b>: {:.3f} <br><b>FAH</b>: {:.3f}'.format(
                    row['_rxn_M_inorganic'], row['_rxn_M_organic'], row['_rxn_M_acid']) for idx, row in df.iterrows()],
                hoverinfo='text',
                marker=dict(
                    size=4,
                    color=self.trace_colors[i],
                    line=dict(
                        width=0.2
                    ),
                    opacity=1.0
                )
            )
            self.amine_crystal_traces.append(trace)
            if i == 3:
                # normed = preprocessing.minmax_scale(
                #    [df['_rxn_M_inorganic'], df['_rxn_M_organic'], df['_rxn_M_acid']], axis=1)

                # normed_success_points = np.dstack(
                #    (normed[0], normed[1], normed[2]))[0]
                success_points = np.dstack(
                    (df['_rxn_M_inorganic'], df['_rxn_M_organic'], df['_rxn_M_acid']))[0]

                ones_hull = None
                if self.clustering is not None:
                    self.clustering.fit(success_points)
                    # print(self.clustering.labels_)
                    one_labels = success_points[[
                        True if i == 0 else False for i in self.clustering.labels_]]
                    ones_hull = ConvexHull(one_labels)

                success_hull = None
                if len(success_points):
                    success_hull = ConvexHull(success_points)

        self.data = self.amine_crystal_traces

        if ones_hull is not None:
            xp, yp, zp = zip(*one_labels[ones_hull.vertices])
            self.ones_hull_plot = go.Mesh3d(x=xp,
                                            y=yp,
                                            z=zp,
                                            color='yellow',
                                            opacity=0.50,
                                            alphahull=0)

            x_mean = np.mean(xp)
            y_mean = np.mean(yp)
            z_mean = np.mean(zp)

            # print('Cluster Hull centroid wrt points: ({}, {}, {})'.format(
            #    x_mean, y_mean, z_mean))
            # print('Cluster Hull centroid wrt sides: ({}, {}, {})'.format(
            #    *self.com_edges(xp, yp, zp)))
            self.data += [self.ones_hull_plot]

        if success_hull:
            xp, yp, zp = zip(*success_points[success_hull.vertices])
            self.success_hull_plot = go.Mesh3d(x=xp,
                                               y=yp,
                                               z=zp,
                                               color='red',
                                               opacity=0.50,
                                               alphahull=0)

            x_mean = np.mean(xp)
            y_mean = np.mean(yp)
            z_mean = np.mean(zp)

            # print('Success Hull centroid wrt points: ({}, {}, {})'.format(
            #    x_mean, y_mean, z_mean))
            # print('Success Hull centroid wrt sides: ({}, {}, {})'.format(
            #    *self.com_edges(xp, yp, zp)))
            self.data += [self.success_hull_plot]

        if self.hull_mesh:
            self.data += [self.hull_mesh]

    def com_edges(self, px, py, pz):
        sx = sy = sz = slen = 0
        x1 = px[-1]
        y1 = py[-1]
        z1 = pz[-1]
        for i in range(len(px)):
            x2 = px[i]
            y2 = py[i]
            z2 = pz[i]
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            length = math.sqrt(dx*dx + dy*dy + dz*dz)
            sx = sx + (x1 + x2)/2*length
            sy = sy + (y1 + y2)/2*length
            sz = sz + (z1 + z2)/2*length
            slen = slen + length
            x1 = x2
            y1 = y2
            z1 = z2
        cx = sx/slen
        cy = sy/slen
        cz = sz/slen
        return cx, cy, cz

    def setup_plot(self, xaxis_label='Lead Iodide [PbI3] (M)',
                   yaxis_label='Dimethylammonium Iodide<br>[Me2NH2I] (M)',
                   zaxis_label='Formic Acid [FAH] (M)'):
        self.layout = go.Layout(
            scene=dict(
                xaxis=dict(
                    title=xaxis_label,
                    tickmode='linear',
                    dtick=0.5,
                    range=[0, self.max_inorg],
                ),
                yaxis=dict(
                    title=yaxis_label,
                    tickmode='linear',
                    dtick=0.5,
                    range=[0, self.max_org],
                ),
                zaxis=dict(
                    title=zaxis_label,
                    tickmode='linear',
                    dtick=1.0,
                    range=[0, self.max_acid],
                ),
            ),
            legend=go.layout.Legend(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2
            ),
            width=975,
            height=700,
            margin=go.layout.Margin(
                l=20,
                r=20,
                b=20,
                t=20,
                pad=2
            ),
        )
        try:
            with self.fig.batch_update():
                for i, trace in enumerate(self.data):
                    self.fig.data[i].x = trace.x
                    self.fig.data[i].y = trace.y
                    self.fig.data[i].z = trace.z
                self.fig.layout.update(self.layout)
        except:
            self.fig = go.FigureWidget(data=self.data, layout=self.layout)
            for trace in self.fig.data[:-1]:
                trace.on_click(self.show_data_3d_callback)

    def setup_widgets(self, image_folder='images'):
        image_folder = self.base_path + '/' + image_folder
        self.image_list = os.listdir(image_folder)

        # Data Filter Setup
        download_robot_file = Button(
            description='Download Robot File',
            disabled=False,
            button_style='',
            tooltip='Click to download robot file of the current plate',
        )

        download_prep_file = Button(
            description='Download Reagent Prep',
            disabled=False,
            button_style='',
            tooltip='Click to download reagent preperation file for the current plate',
        )

        reset_plot = Button(
            description='Reset',
            disabled=False,
            tooltip='Reset the colors of the plot'
        )

        xy_check = Button(
            description='Show X-Y axes',
            disabled=False,
            button_style='',
            tooltip='Click to show X-Y axes'
        )

        show_hull_check = ToggleButton(
            value=True,
            description='Show State Space',
            disabled=False,
            button_style='',
            tooltip='Toggle to show/hide state space',
            icon='check'
        )

        show_success_hull = ToggleButton(
            value=True,
            description='Show sucess hull',
            disabled=False,
            button_style='',
            tooltip='Toggle to show/hide success hull',
            icon='check'
        )

        self.select_amine = Dropdown(
            options=[row[1]['Chemical Name'] for row in self.organic_inchis.iterrows(
            ) if len(self.full_perovskite_data.loc[self.full_perovskite_data['_rxn_organic-inchikey'] == row[1]['InChI Key (ID)']]) > 0],
            description='Amine:',
            disabled=False,
        )

        self.select_solvent = Dropdown(
            options=[None],
            description='Solvent:',
            disabled=False,
        )

        download_robot_file.on_click(self.download_robot_callback)
        download_prep_file.on_click(self.download_prep_callback)
        reset_plot.on_click(self.reset_plot_callback)
        xy_check.on_click(self.set_xy_camera)
        show_hull_check.observe(self.toggle_mesh, 'value')
        show_success_hull.observe(self.toggle_success_mesh, 'value')
        self.select_amine.observe(self.select_amine_callback, 'value')
        self.select_solvent.observe(self.select_solvent_callback, 'value')

        # Experiment data tab setup
        self.experiment_table = HTML()
        self.experiment_table.value = "Please click on a point to explore experiment details"

        self.image_data = {}
        for img_filename in os.listdir(image_folder):
            with open("{}/{}".format(image_folder, img_filename), "rb") as f:
                b = f.read()
                self.image_data[img_filename] = b

        self.image_widget = Image(
            value=self.image_data['not_found.png'],
            layout=Layout(height='400px', width='650px')
        )

        experiment_view_vbox = VBox(
            [HBox([self.experiment_table, self.image_widget])])

        plot_tabs = Tab([VBox([self.fig,
                               HBox([self.select_amine, self.select_solvent]),
                               HBox([xy_check, show_hull_check, show_success_hull, reset_plot])]),
                         ])
        plot_tabs.set_title(0, 'Chemical Space')

        self.full_widget = VBox([plot_tabs, experiment_view_vbox])
        self.full_widget.layout.align_items = 'center'

    def select_amine_callback(self, state):
        new_amine_name = state['new']
        new_amine_inchi = self.organic_inchis[self.organic_inchis['Chemical Name']
                                              == new_amine_name].iloc[0]['InChI Key (ID)']
        amine_data = self.full_perovskite_data[self.full_perovskite_data['_rxn_organic-inchikey'] == new_amine_inchi]
        solvents = [
            inchi for inchi in amine_data['_raw_reagent_0_chemicals_0_InChIKey'].unique()]

        self.select_solvent.options = ['All'] + [
            key for key, value in self.solvent_inchis.items() if value in solvents]

        solvent_inchi = ""
        self.generate_plot(new_amine_inchi, solvent_inchi)

    def select_solvent_callback(self, state):
        new_solvent_inchi = self.solvent_inchis[state['new']]

        amine = self.select_amine.value
        amine_inchi = self.organic_inchis[self.organic_inchis['Chemical Name']
                                          == amine].iloc[0]['InChI Key (ID)']
        self.generate_plot(amine_inchi, new_solvent_inchi)

    def get_plate_options(self):
        plates = set()
        for df in self.amine_crystal_dfs:
            for i, row in df.iterrows():
                name = str(row['name'])
                plate_name = '_'.join(name.split('_')[:-1])
                plates.add(plate_name)
        plate_options = []
        for i, plate in enumerate(plates):
            plate_options.append(plate)
        return plate_options

    def generate_table(self, row, columns, column_names):
        table_html = """ <table border="1" style="width:100%;">
                        <tbody>"""
        for i, column in enumerate(columns):
            if isinstance(row[column], str):
                value = row[column].split('_')[-1]
            else:
                value = np.round(row[column], decimals=3)
            table_html += """
                            <tr>
                                <td style="padding: 8px;">{}</td>
                                <td style="padding: 8px;">{}</td>
                            </tr>
                          """.format(column_names[i], value)
        table_html += """
                        </tbody>
                        </table>
                        """
        return table_html

    def toggle_mesh(self, state):
        with self.fig.batch_update():
            self.fig.data[-1].visible = state.new

    def toggle_success_mesh(self, state):
        with self.fig.batch_update():
            self.fig.data[-2].visible = state.new

    def set_xy_camera(self, state):
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=0.0, z=2.5)
        )

        self.fig['layout'].update(
            scene=dict(camera=camera),
        )

    def download_robot_callback(self, b):
        if self.selected_plate:
            js_string = """
                        var base_uri = utils.get_body_data("baseUrl");
                        var file_location = utils.encode_uri_components('{}_RobotInput.xls');
                        // var final_url = utils.url_path_join(base_uri, 'files', file_location)
                        var final_url = 'files/' + file_location
                        console.log(final_url)
                        window.open(final_url, IPython._target);
                        """.format(self.selected_plate)

            display(Javascript(js_string))

    def download_prep_callback(self, b):
        if self.selected_plate:
            js_string = """
                        var base_uri = utils.get_body_data("baseUrl");
                        var file_location = utils.encode_uri_components('{}_ExpDataEntry.xlsx');
                        // var final_url = utils.url_path_join(base_uri, 'files', file_location)
                        var final_url = 'files/' + file_location
                        console.log(final_url)
                        window.open(final_url, IPython._target);
                        """.format(self.selected_plate)

            display(Javascript(js_string))

    def reset_plot_callback(self, b):
        with self.fig.batch_update():
            for i in range(len(self.fig.data[:4])):
                self.fig.data[i].marker.color = self.trace_colors[i]
                self.fig.data[i].marker.size = 4

    def show_data_3d_callback(self, trace, point, selector):
        if point.point_inds and point.trace_index < 4:

            selected_experiment = self.amine_crystal_dfs[point.trace_index].iloc[point.point_inds[0]]
            with self.fig.batch_update():
                for i in range(len(self.fig.data[:4])):
                    color = self.trace_colors[i].split(',')
                    color[-1] = '0.5)'
                    color = ','.join(color)
                    if i == point.trace_index:
                        marker_colors = [color for x in range(len(trace['x']))]
                        marker_colors[point.point_inds[0]
                                      ] = self.trace_colors[i]
                        self.fig.data[i].marker.color = marker_colors
                        self.fig.data[i].marker.size = 6
                    else:
                        self.fig.data[i].marker.color = color
                        self.fig.data[i].marker.size = 4
            self.populate_data(selected_experiment)

    def populate_data(self, selected_experiment):
        name = selected_experiment['name']
        if name+'.jpg' in self.image_list:
            self.image_widget.value = self.image_data[name+'.jpg']
        else:
            self.image_widget.value = self.image_data['not_found.png']
        columns = ['name', '_rxn_M_acid', '_rxn_M_inorganic', '_rxn_M_organic',
                   '_rxn_mixingtime1S', '_rxn_mixingtime2S', '_rxn_reactiontimeS',
                   '_rxn_stirrateRPM', '_rxn_temperatureC_actual_bulk']
        column_names = ['Well ID', 'Formic Acid [FAH]', 'Lead Iodide [PbI2]', 'Dimethylammonium Iodide [Me2NH2I]',
                        'Mixing Time Stage 1 (s)', 'Mixing Time Stage 2 (s)', 'Reaction Time (s)',
                        'Stir Rate (RPM)', 'Temperature (C)']

        prefix = '_'.join(name.split('_')[:-1])
        self.selected_plate = prefix
        self.experiment_table.value = '<p>Plate ID:<br> {}</p>'.format(prefix) + self.generate_table(
            selected_experiment.loc[columns], columns, column_names)

    @property
    def plot(self):
        return self.full_widget
