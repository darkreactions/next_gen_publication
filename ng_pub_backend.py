import os
from ipywidgets import Tab, SelectMultiple, Accordion, ToggleButton, VBox, HBox, HTML, Image, Button, Text, Dropdown
from _plotly_future_ import v4_subplots
import plotly as py
import plotly.graph_objs as go
from ipywidgets import HBox, VBox, Image, Layout, HTML
import numpy as np
import pandas as pd
from IPython.display import display, Javascript, FileLink
from notebook import notebookapp
import urllib.parse


class Figure1:
    def __init__(self, csv_file_path, base_path=''):
        self.selected_plate = None
        self.old_plate = None
        self.base_path = base_path
        self.full_perovskite_data = pd.read_csv(
            csv_file_path, low_memory=False, skiprows=4)
        self.setup_hull()
        self.gen_amine_traces()
        self.setup_plot()
        self.generate_xrd()
        self.setup_widgets()

    def setup_hull(self, hull_points=[[0., 0., 0.], [0., 2.3, 0.], [1.86, 1.86, 0.],
                                      [0., 0., 9.5], [1.19339, 1.19339, 9.5], [0., 1.4757, 9.5]]):
        xp, yp, zp = zip(*hull_points)
        self.hull_mesh = go.Mesh3d(x=xp,
                                   y=yp,
                                   z=zp,
                                   color='#FFB6C1',
                                   opacity=0.50,
                                   alphahull=0)

    def gen_amine_traces(self, inchi_key='JMXLWMIFDJCGBV-UHFFFAOYSA-N', amine_short_name='Me2NH2I'):
        amine_data = self.full_perovskite_data.loc[self.full_perovskite_data['_rxn_organic-inchikey'] == inchi_key]
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
        self.data = self.amine_crystal_traces

        if self.hull_mesh:
            self.data += [self.hull_mesh]

    def setup_plot(self, xaxis_label='Lead Iodide [PbI3] (M)',
                   yaxis_label='Dimethylammonium Iodide<br>[Me2NH2I] (M)',
                   zaxis_label='Formic Acid [FAH] (M)'):
        self.layout = go.Layout(
            scene=dict(
                xaxis=dict(
                    title=xaxis_label,
                    tickmode='linear',
                    dtick=0.5,
                    range=[0, 2.0],
                ),
                yaxis=dict(
                    title=yaxis_label,
                    tickmode='linear',
                    dtick=0.5,
                    range=[0, 2.5],
                ),
                zaxis=dict(
                    title=zaxis_label,
                    tickmode='linear',
                    dtick=1.0,
                    range=[0, 9.5],
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
            description='Show Convex Hull',
            disabled=False,
            button_style='',
            tooltip='Toggle to show/hide convex hull',
            icon='check'
        )

        download_robot_file.on_click(self.download_robot_callback)
        download_prep_file.on_click(self.download_prep_callback)
        reset_plot.on_click(self.reset_plot_callback)
        xy_check.on_click(self.set_xy_camera)
        show_hull_check.observe(self.toggle_mesh, 'value')

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

        self.thermal_plot = self.init_thermal_plot()
        plate_options = self.get_plate_options()
        self.selected_plate = plate_options[0]
        self.generate_thermal(self.selected_plate)
        self.plate_dropdown = Dropdown(options=plate_options,
                                       description='Plate:',
                                       )
        self.plate_dropdown.observe(self.change_plates, 'value')

        #tab = Tab()
        #tab.children = [experiment_view_vbox]
        #tab.set_title(0, 'Experiment Details')
        #tab.set_title(1, 'XRD data')

        plot_tabs = Tab([VBox([self.fig,
                               HBox([xy_check, show_hull_check, reset_plot])]),
                         VBox([self.thermal_plot,
                               HBox([self.plate_dropdown, download_robot_file, download_prep_file])]),
                         ])
        plot_tabs.set_title(0, 'Chemical Space')
        plot_tabs.set_title(1, 'Plate')

        self.full_widget = VBox([plot_tabs, experiment_view_vbox])
        self.full_widget.layout.align_items = 'center'

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

    def change_plates(self, state):
        self.selected_plate = state.new
        self.generate_thermal(self.selected_plate)

    def init_thermal_plot(self):
        import plotly.graph_objs as go

        #base_url = list(notebookapp.list_running_servers())[0]['base_url']
        #data_url = base_url + 'files/sd2e-community/perovskite-data/data_connected_pub'
        full_url = 'images/not_found.png'
        layout = go.Layout(
            hovermode='closest',
            xaxis=dict(
                title='Column',
                showgrid=False,
                ticktext=[i+1 for i in range(12)],
                tickvals=[i*1.0+0.5 for i in range(12)],

            ),
            yaxis=dict(
                title='Row',
                showgrid=False,
                ticktext=list('ABCDEFGH'),
                tickvals=[i*1.0+0.5 for i in range(8)],

            ),
            images=[
                go.layout.Image(
                    source=full_url,
                    xref="x",
                    yref="y",
                    x=0,
                    y=8,
                    sizex=12,
                    sizey=8,
                    sizing="stretch",
                    opacity=0.8,
                    layer="below")
            ],

            width=950,
            height=700,

        )
        from itertools import product
        xy = list(product([i*1.0+0.5 for i in range(12)],
                          [i*1.0+0.5 for i in range(8)]))
        x, y = list(zip(*xy))
        self.vial_labels = list(product([i+1 for i in range(12)],
                                        list('ABCDEFGH')))
        trace1 = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(size=45,
                        color='rgba(66, 245, 87, 0.1)',
                        line=dict(width=3,
                                  color='rgb(0, 0, 0)')),
            opacity=1.0,
        )
        fig = go.FigureWidget(data=[trace1], layout=layout)
        fig.data[0].on_click(self.show_data_thermal_callback)
        return fig

    def generate_xrd(self):
        import pandas as pd
        dat = pd.read_csv(self.base_path +
                          '/2019-02-21T16_54_38.714239+00_00_LBL_1.xy', skiprows=1, delimiter=' ')
        dat.columns = ['x', 'y']
        trace = go.Scatter(
            x=dat['x'],
            y=dat['y'],
        )
        data = [trace]
        layout = go.Layout(
            autosize=False,
            width=1000,
            height=600,
            margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
        )
        self.xrd_plot = go.FigureWidget(data=data, layout=layout)

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
            for i in range(len(self.fig.data[:-1])):
                self.fig.data[i].marker.color = self.trace_colors[i]
                self.fig.data[i].marker.size = 4

    def generate_thermal(self, new_plate):
        if new_plate != self.old_plate:
            self.old_plate = new_plate
            with self.thermal_plot.batch_update():
                # base_url = list(notebookapp.list_running_servers())[
                #    0]['base_url']
                #data_url = base_url + 'files/sd2e-community/perovskite-data/data_connected_pub'
                full_url = 'images/{}_thermal.PNG'.format(new_plate)
                self.thermal_plot.layout.images = [go.layout.Image(
                    source=full_url,
                    xref="x",
                    yref="y",
                    x=0,
                    y=8,
                    sizex=12,
                    sizey=8,
                    sizing="stretch",
                    opacity=0.8,
                    layer="below")]

    def show_data_3d_callback(self, trace, point, selector):
        if point.point_inds:
            selected_experiment = self.amine_crystal_dfs[point.trace_index].iloc[point.point_inds[0]]
            with self.fig.batch_update():
                for i in range(len(self.fig.data[:-1])):
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
            # Code to update thermal image
            self.generate_thermal(self.selected_plate)

    def show_data_thermal_callback(self, trace, point, selector):
        pt_idx = point.point_inds[0]
        vial_name = self.vial_labels[pt_idx][1] + \
            str(self.vial_labels[pt_idx][0])
        experiment_name = self.selected_plate+'_'+vial_name
        #self.amine_crystal_dfs['preTestScore'].where(df['postTestScore'] > 50)
        selected_experiment = None
        for df in self.amine_crystal_dfs:
            if not df.loc[df['name'] == experiment_name].empty:
                selected_experiment = df.loc[df['name']
                                             == experiment_name].iloc[0]
                # print(selected_experiment['name'])
                self.populate_data(selected_experiment)
                break
        marker_colors = [
            'rgba(66, 245, 87, 0.1)' for i in range(len(trace['x']))]
        score = int(selected_experiment['_out_crystalscore'] - 1)
        marker_colors[pt_idx] = self.trace_colors[score]
        with self.thermal_plot.batch_update():
            self.thermal_plot.data[0].marker.color = marker_colors

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


class XYPlot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def plot(self, xaxis_label, yaxis_label, annotations=[], reversed=False, extra_traces=None):
        ann = []
        for a in annotations:
            ann.append(go.layout.Annotation(
                x=a[0],
                y=a[1],
                xref="x",
                yref="y",
                text=a[2],
                showarrow=False,
                ax=20,
                ay=-30,
            ))
        trace = go.Scatter(
            x=self.x,
            y=self.y,
        )

        data = [trace]
        if extra_traces:
            for t in extra_traces:
                data.append(t)

        layout = go.Layout(
            showlegend=False,
            annotations=ann,
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text=xaxis_label,
                ),
                autorange='reversed' if reversed else True,
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text=yaxis_label,
                )
            ),
            autosize=False,
            width=1000,
            height=600,
            margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
        )
        return go.FigureWidget(data=data, layout=layout)


class JsMolFigure:
    def __init__(self, cif_paths, fig_names, doi_values, widget_side=400):
        self.cif_paths = cif_paths
        self.fig_names = fig_names
        self.doi_values = doi_values
        self.widget_side = widget_side
        base_url = list(notebookapp.list_running_servers())[0]['base_url']
        self.data_url = ''
        self.html = """
                    <!doctype html>
                    <html>
                    <head>
                    <meta content="text/html; charset=UTF-8" http-equiv="content-type">
                    <script
                        src="https://code.jquery.com/jquery-3.4.1.min.js"
                        integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo="
                        crossorigin="anonymous"></script>

                    <script type="text/javascript" src="jsmol/JSmol.min.js"></script>
                    <style type="text/css">
                    .plot_div{{
                        display: inline;
                        margin: 15px;
                        float: left;
                    }}
                    </style>
                    """.format(self.data_url)
        jmol_html = ''
        jmol_script = ''
        for i, cif_path in enumerate(cif_paths):

            jmol_script += """
                    <!-- CSS Style Inline: -->
                    <style type="text/css">
                        /* Jmol Applet */
                        /* defines height, width and orientation of the Jmol div element */
                        #jmol_div_{1}{{
                            height: {5}px;
                            width:  {5}px;
                            margin: 5px;
                        }}
                    </style>
                    <!-- calls to jQuery and Jmol (inline) -->
                    <script type="text/javascript">
                        // Jmol readyFunction
                        // is called when Jmol is ready

                        jmol_isReady_{1} = function(applet) {{
                            Jmol._getElement(applet, "appletdiv").style.border="1px solid blue";
                        }}
                        // initialize Jmol Applet
                        var myJmol_{1} = "myJmol_{1}";
                        var Info_{1} = {{
                            width:   "100%",
                            height:  "100%",
                            color:   "#000000", //black
                            use:     "HTML5",
                            j2sPath: "./jsmol/j2s", 
                            jarPath: "./jsmol/java",
                            jarFile: "JmolAppletSigned.jar",
                            debug:   false,
                            readyFunction: jmol_isReady_{1},
                            script:  'load "{0}" ; hide _H;',
                            //script: 'load ":tylenol";',
                            allowJavaScript: true,
                            disableJ2SLoadMonitor: true,
                        }}
                        // jQuery ready functions
                        // is called when page has been completely loaded
                        $(document).ready(function() {{
                            console.log('Document is ready');
                            
                        }} )
                        function populate_jmol_{1}(){{
                            html_text = "<center><h3>{4}</h3></center>" + Jmol.getAppletHtml(myJmol_{1}, Info_{1});
                            html_text += Jmol.jmolButton(myJmol_{1},'load "{0}"  SUPERCELL {{2 2 2}}; hide _H;', "Show 2x2x2 supercell");
                            html_text += Jmol.jmolButton(myJmol_{1},'load "{0}"; hide _H;', "Reset");
                            html_text += '<a href="{2}" target="_blank"><button> Download {4} CIF</button></a>'
                            $("#jmol_div_{1}").html(html_text);
                        }}
                        var lastPrompt=0;
                    </script>
                    """.format(cif_path, i, self.link_generator(self.fig_names[i], i), self.data_url, self.fig_names[i], self.widget_side)

            jmol_html += """
                        <div class='plot_div'>
                            <div id='jmol_div_{0}'> 
                                <div style="text-align: center;">
                                    <button type="button" onclick=populate_jmol_{0}()>Show Molecule</button>
                                </div> 
                            </div>
                        </div>
                        """.format(i)
        self.html += jmol_script + "</head> <body>" + jmol_html + "</body> </html>"

    def link_generator(self, fig_name, i):
        if 'HUTVAV' not in fig_name:
            return self.data_url + '/' + self.cif_paths[i]
        else:
            # return 'https://www.ccdc.cam.ac.uk/structures/Search?Ccdcid={}&DatabaseToSearch=Published'.format(fig_name)
            return 'https://www.ccdc.cam.ac.uk/structures/Search?Doi={}&DatabaseToSearch=Published'.format(urllib.parse.quote(self.doi_values[fig_name]))

    @property
    def plot(self):
        from IPython.display import HTML
        # from ipywidgets import HTML
        return HTML(self.html)

    @property
    def controls(self):
        get_supercell = ToggleButton(
            value=False,
            description='Get supercell',
            disabled=False,
            button_style='',
            tooltip='Click to show supercell'
        )
        get_supercell.observe(self.supercell_callback, 'value')

        run_command = Button(
            description='Run Command',
            disabled=False
        )
        run_command.on_click(self.run_cmd_callback)

        self.command_text = Text(
            value='spin on',
            placeholder='spin on',
            description='Command:',
            disabled=False
        )

        data_filter_vbox = VBox(
            [HBox([get_supercell]), HBox([self.command_text, run_command])])

        return data_filter_vbox

    def run_cmd_callback(self, b):
        if self.command_text.value:
            js_string = """Jmol.script(myJmol_{}, '{}');""".format(
                self.id, self.command_text.value)
            display(Javascript(js_string))

    def supercell_callback(self, state):
        if state.new:
            js_string = """Jmol.script(myJmol_{0}, 'load "{1}" supercell {{2 2 2}}');""".format(
                self.id, self.cif_path)
        else:
            js_string = """Jmol.script(myJmol_{0}, 'load "{1}"');""".format(
                self.id, self.cif_path)
        display(Javascript(js_string))


class MLSection:
    def __init__(self, data_path, inchi_key='JMXLWMIFDJCGBV-UHFFFAOYSA-N'):
        from collections import defaultdict
        self.inchi_key = inchi_key
        full_perovskite_data = pd.read_csv(
            data_path, low_memory=False, skiprows=4)
        self._dataframe = full_perovskite_data.loc[full_perovskite_data['_rxn_organic-inchikey'] == inchi_key]
        self._models = []
        self.descriptor_columns = [col for col in self._dataframe.columns if (
            'rxn' in col or 'feat' in col)]
        self.descriptors = self._dataframe[self.descriptor_columns].drop(
            '_rxn_organic-inchikey', axis=1)
        self.crystal_scores = self._dataframe['_out_crystalscore']

        # Metrics
        self.metrics = {}
        self.metrics['accuracy'] = defaultdict(list)
        self.metrics['accuracy_std'] = defaultdict(list)
        self.metrics['precision'] = defaultdict(list)
        self.metrics['precision_std'] = defaultdict(list)
        self.metrics['F1'] = defaultdict(list)
        self.metrics['F1_std'] = defaultdict(list)
        self.metrics['recall'] = defaultdict(list)
        self.metrics['recall_std'] = defaultdict(list)

    @property
    def dataframe(self):
        return self._dataframe

    def generate_split(self, folds=5, test_size=0.2):
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.preprocessing import StandardScaler

        _descriptors_vector = StandardScaler().fit_transform(self.descriptors)
        _crystal_score_vector = self.crystal_scores.replace(
            [1, 2, 3], 0).replace([4], 1).values

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        sss = StratifiedShuffleSplit(
            n_splits=folds, test_size=test_size, random_state=42)
        for train_index, test_index in sss.split(_descriptors_vector, _crystal_score_vector):
            X_train.append(_descriptors_vector[train_index])
            y_train.append(_crystal_score_vector[train_index])
            X_test.append(_descriptors_vector[test_index])
            y_test.append(_crystal_score_vector[test_index])

        return (X_train, y_train, X_test, y_test)

    def add_model(self, model):
        self._models.append(model)

    def run_models(self, learn_rate=[0.02, 0.06, 0.1, 0.33, 0.55, 0.78], n_splits=5):
        from sklearn.metrics import precision_score, recall_score, f1_score
        self.learn_rate = learn_rate
        for rate in self.learn_rate:
            X_train, y_train, X_test, y_test = self.generate_split(
                folds=n_splits, test_size=1-rate)
            for model_num, model in enumerate(self._models):
                model_name = type(model).__name__ + '_' + str(model_num)
                acc_list = []
                precision_list = []
                recall_list = []
                f1_list = []
                for i in range(n_splits):
                    model.fit(X_train[i], y_train[i])
                    y_predict = model.predict(X_test[i])
                    acc_list.append(model.score(X_test[i], y_test[i]))
                    precision_list.append(
                        precision_score(y_test[i], y_predict))
                    recall_list.append(recall_score(y_test[i], y_predict))
                    f1_list.append(f1_score(y_test[i], y_predict))
                self.metrics['accuracy'][model_name].append(np.mean(acc_list))
                self.metrics['accuracy_std'][model_name].append(
                    np.std(acc_list))
                self.metrics['precision'][model_name].append(
                    np.mean(precision_list))
                self.metrics['precision_std'][model_name].append(
                    np.std(precision_list))
                self.metrics['F1'][model_name].append(np.mean(f1_list))
                self.metrics['F1_std'][model_name].append(
                    np.std(f1_list))
                self.metrics['recall'][model_name].append(np.mean(recall_list))
                self.metrics['recall_std'][model_name].append(
                    np.std(recall_list))

    def interpret_complex_model(self, data_features, predicted_labels, interpretable_model=None):
        # make interpretable model a decision tree by default
        if interpretable_model == None:
            from sklearn.tree import DecisionTreeClassifier
            interpretable_model = DecisionTreeClassifier(random_state=0)

        # fit interpretable model on complex model's predicted labels
        # NOTE: random_state param has to be passed to get consistent tree output, but is an arbitrary number
        # How do we know which tree is the 'correct' way that the UninterpretableModel is learning?
        interpretable_model.fit(data_features, predicted_labels)

        # visualize model
        if isinstance(interpretable_model, DecisionTreeClassifier):
            from sklearn.externals.six import StringIO
            from IPython.display import Image
            from sklearn.tree import export_graphviz
            import pydotplus
            dot_data = StringIO()
            export_graphviz(interpretable_model, out_file=dot_data,
                            filled=True, rounded=True,
                            special_characters=True,
                            feature_names=list(data_features.columns))
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

            return Image(graph.create_png())

    def plot(self, metric='accuracy'):
        color_list = ['rgba(31, 118, 180, 1)', 'rgba(255, 127, 14, 1)', 'rgba(44, 160, 44, 1)', 'rgba(214, 39, 39, 1)',
                      'rgba(147, 103, 189, 1)', 'rgba(140, 86, 75, 1)', 'rgba(227, 119, 195, 1)', 'rgba(127, 127, 127, 1)',
                      'rgba(189, 189, 34, 1)', 'rgba(23, 189, 207, 1)']
        data = self.metrics[metric]
        std_dev = self.metrics[metric+'_std']
        yaxis_label = metric

        trace_list = []
        layout = go.Layout(
            hovermode='closest',
            showlegend=True,
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text="Number of training experiments",
                ),
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text=yaxis_label,
                )
            ),
        )
        for i, model in enumerate(data):
            x = [x*len(self.descriptors) for x in self.learn_rate]
            y = data[model]
            trace = go.Scatter(
                name=model,
                x=x,
                y=y,
                marker=dict(size=5, color=color_list[i % len(color_list)],),
                opacity=1.0,

            )
            trace_list.append(trace)
            if std_dev[model]:
                import numpy as np
                x_rev = x[::-1]
                y_upper = np.array(data[model]) + np.array(std_dev[model])
                y_lower = np.array(data[model]) - np.array(std_dev[model])
                trace2 = go.Scatter(
                    x=x+x_rev,
                    y=list(y_upper)+list(y_lower)[::-1],
                    fill='tozerox',
                    fillcolor=self.change_alpha(
                        color_list[i % len(color_list)], 0.3),
                    line=dict(color='rgba(255,255,255,0)'),
                    name=model,
                    showlegend=False,
                )
                trace_list.append(trace2)

        return go.FigureWidget(data=trace_list, layout=layout)

    def change_alpha(self, color, alpha):
        color = color.split(',')
        color[-1] = '{})'.format(alpha)
        return ','.join(color)
