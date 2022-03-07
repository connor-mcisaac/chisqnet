import os
from chisqnet.preprocessing import TriggerList
import pycbc.workflow as wf
from pycbc.workflow.core import Executable, Node
from pycbc.workflow.jobsetup import (PyCBCInspiralExecutable, int_gps_time_to_str,
                                     identify_needed_data, JobSegmenter)


class FilterTriggersExecutable(Executable):

    current_retention_level = Executable.ALL_TRIGGERS
    def create_node(self, single_det_files, bank_file,
                    injfind_file, veto_file):
        node = Node(self)
        node.add_input_list_opt('--trigger-files', single_det_files)
        if injfind_file is not None:
            node.add_input_opt('--injfind-file', injfind_file)
        node.add_input_opt('--bank-file', bank_file)
        if veto_file is not None:
            node.add_input_opt('--veto-files', veto_file)
        seg = single_det_files.get_times_covered_by_files()
        node.new_output_file_opt(seg, '.hdf', '--output-file')
        return node


class PreprocessingDaxGenerator(Executable):
    """
    This runs make_preprocessing_workflow and creates a sub-workflow
    the output file is therefore dealt with differently
    Based on: https://github.com/gwastro/pycbc/blob/master/bin/workflows/pycbc_make_uberbank_workflow#L93
    """

    current_retention_level = Executable.FINAL_RESULT
    file_input_options = []
    def create_node(self, analysis_time, config_file, subsections, out_storage_path,
                    output_file, trigger_file, injection_file=None,
                    workflow_name='preprocessing_workflow', sub_wf_tags=None):
        node = Node(self)

        node.add_input_opt('--config-files', config_file)
        
        # Define the ouput file of the workflow as an option
        # as this will not be created by this job itself
        node.add_opt('--output-file', output_file.storage_path)
        
        node.add_opt('--workflow-name', workflow_name)
        node.add_opt('--output-dir', out_storage_path)
        node.add_opt('--is-sub-workflow','')

        node.add_input_opt('--trigger-file', trigger_file)
        if injection_file is not None:
            node.add_input_opt('--injection-file', injection_file)

        if sub_wf_tags is not None:
            node.add_opt('--tags', ' '.join(sub_wf_tags))
        else:
            sub_wf_tags = []

        config_removes = []
        for section in subsections:
            config_removes.append('workflow-filter-' + section)

        config_overrides = []
        config_overrides.append('workflow:start-time:{0}'.format(analysis_time[0]))
        config_overrides.append('workflow:end-time:{0}'.format(analysis_time[1]))

        node.add_opt('--config-delete', ' '.join(config_removes))
        node.add_opt('--config-overrides', ' '.join(config_overrides))

        node.new_output_file_opt(analysis_time, '.dax',
                                 '--dax-filename',
                                 tags=self.tags + sub_wf_tags + ['DAX'])
        node.new_output_file_opt(analysis_time, '.map',
                                 '--map-filename',
                                 tags=self.tags + sub_wf_tags + ['MAP'])
        node.new_output_file_opt(analysis_time, '.tc.txt',
                                 '--transformation-catalog-filename',
                                 tags=self.tags + sub_wf_tags +
                                 ['TRANSFORMATION_CATALOG'])

        return node


class PreprocessStrainExecutable(PyCBCInspiralExecutable):
    """ Modified vesion of the class PyCBCInspiralExecutable
    Used to preprocess strain for training making sure it is
    generated in the same way as PyCBC"""

    current_retention_level = Executable.ALL_TRIGGERS
    file_input_options = ['--gating-file']
    time_dependent_options = ['--channel-name']

    def __init__(self, cp, exe_name, trigger_file, ifo=None, out_dir=None,
                 injection_file=None, tags=None, reuse_executable=False):

        super(PreprocessStrainExecutable, self).__init__(
            cp,
            exe_name,
            ifo=ifo,
            out_dir=out_dir,
            injection_file=injection_file,
            tags=tags,
            reuse_executable=reuse_executable
        )
        self.trigger_file = trigger_file

    def create_node(self, data_seg, valid_seg, dfParents=None, tags=None):
        if tags is None:
            tags = []
        node = Node(self, valid_seg=valid_seg)
        if not self.has_opt('pad-data'):
            raise ValueError("The option pad-data is a required option of "
                             "%s. Please check the ini file." % self.name)
        pad_data = int(self.get_opt('pad-data'))

        # set remaining options flags
        node.add_opt('--gps-start-time',
                     int_gps_time_to_str(data_seg[0] + pad_data))
        node.add_opt('--gps-end-time',
                     int_gps_time_to_str(data_seg[1] - pad_data))
        node.add_opt('--trig-start-time', int_gps_time_to_str(valid_seg[0]))
        node.add_opt('--trig-end-time', int_gps_time_to_str(valid_seg[1]))

        node.add_opt('--ifo', self.ifo_list[0])
        node.add_input_opt('--trigger-file', self.trigger_file)
        if self.injection_file is not None:
            node.add_input_opt('--injection-file', self.injection_file)

        # set the input and output files
        fil = node.new_output_file_opt(valid_seg, self.ext, '--output', tags=tags,
                         store_file=self.retain_files, use_tmp_subdirs=True)
        fil.add_metadata('data_seg', data_seg)

        if dfParents is not None:
            node.add_input_list_opt('--frame-files', dfParents)

        return node


def sngl_ifo_job_setup(workflow, ifo, out_files, curr_exe_job, science_segs,
                       datafind_outs, trigger_file, allow_overlap=True):
    """ Taken from https://github.com/gwastro/pycbc/blob/0489f33703db64097dc13a0d2385c9ecad913202/pycbc/workflow/jobsetup.py#L180
    edited so that we can supply a trigger file and it will check if there are any triggers in each segment.
    If there are no triggers in a segment then skip it and do not create a job.


    This function sets up a set of single ifo jobs, skipping them if they contain no triggers.
    A basic overview of how this works is as follows:
    * (1) Identify the length of data that each job needs to read in, and what
      part of that data the job is valid for.
    * START LOOPING OVER SCIENCE SEGMENTS
    * (2) Identify how many jobs are needed (if any) to cover the given science
      segment and the time shift between jobs. If no jobs continue.
    * START LOOPING OVER JOBS
    * (3) Identify the time that the given job should produce valid output (ie.
      inspiral triggers) over. If no triggers in this time continue.
    * (4) Identify the data range that the job will need to read in to produce
      the aforementioned valid output.
    * (5) Identify all parents/inputs of the job.
    * (6) Add the job to the workflow
    * END LOOPING OVER JOBS
    * END LOOPING OVER SCIENCE SEGMENTS
    Parameters
    -----------
    workflow: pycbc.workflow.core.Workflow
        An instance of the Workflow class that manages the constructed workflow.
    ifo : string
        The name of the ifo to set up the jobs for
    out_files : pycbc.workflow.core.FileList
        The FileList containing the list of jobs. Jobs will be appended
        to this list, and it does not need to be empty when supplied.
    curr_exe_job : Job
        An instanced of the Job class that has a get_valid times method.
    science_segs : ligo.segments.segmentlist
        The list of times that the jobs should cover
    datafind_outs : pycbc.workflow.core.FileList
        The file list containing the datafind files.
    trigger_file : The parent file to the jobs being setup that contains
        the triggers that are to be analysed.
    allow_overlap : boolean (optional, kwarg, default = True)
        If this is set the times that jobs are valid for will be allowed to
        overlap. This may be desired for template banks which may have some
        overlap in the times they cover. This may not be desired for inspiral
        jobs, where you probably want triggers recorded by jobs to not overlap
        at all.
    Returns
    --------
    out_files : pycbc.workflow.core.FileList
        A list of the files that will be generated by this step in the
        workflow.
    """

    # Load the trigger file
    triggers = TriggerList.read_from_hdf(trigger_file)

    ########### (1) ############
    # Get the times that can be analysed and needed data lengths
    data_length, valid_chunk, valid_length = identify_needed_data(curr_exe_job)

    # Loop over science segments and set up jobs
    for curr_seg in science_segs:
        ########### (2) ############
        # Initialize the class that identifies how many jobs are needed and the
        # shift between them.
        segmenter = JobSegmenter(data_length, valid_chunk, valid_length,
                                 curr_seg, curr_exe_job)

        for job_num in range(segmenter.num_jobs):
            ############## (3) #############
            # Figure out over what times this job will be valid for
            job_valid_seg = segmenter.get_valid_times_for_job(job_num,
                                                   allow_overlap=allow_overlap)

            # If there are no triggers in the valid time skip this job
            triggers_within = triggers.get_time_cut(ifo, job_valid_seg[0], job_valid_seg[1])
            if len(triggers_within['end_time']) == 0:
                continue

            ############## (4) #############
            # Get the data that this job should read in
            job_data_seg = segmenter.get_data_times_for_job(job_num)

            ############# (5) ############
            # Identify parents/inputs to the job

            curr_dfouts = None
            if datafind_outs:
                curr_dfouts = datafind_outs.find_all_output_in_range(ifo,
                                              job_data_seg, useSplitLists=True)
                if not curr_dfouts:
                    err_str = ("No datafind jobs found overlapping %d to %d."
                                %(job_data_seg[0],job_data_seg[1]))
                    err_str += "\nThis shouldn't happen. Contact a developer."
                    raise ValueError(err_str)


            ############## (6) #############
            # Make node and add to workflow

            node = curr_exe_job.create_node(job_data_seg, job_valid_seg,
                                            dfParents=curr_dfouts, tags=[])
            workflow.add_node(node)
            out_files += node.output_files

    return out_files


class MergeSamplesExecutable(Executable):
    
    current_retention_level = Executable.MERGED_TRIGGERS
    def create_node(self, samples, output=None, create_bank=False):
        node = Node(self)
        seg = samples.get_times_covered_by_files()
        node.add_input_list_opt('--sample-files', samples)

        if output is not None:
            # Convert path to file
            out_file = wf.File.from_path(output)
            if self.retain_files:
                if not os.path.isabs(output):
                    out_file.storage_path = os.path.join(self.out_dir,
                                                         output)
                else:
                    out_file.storage_path = output
            node.add_output_opt('--output-file', out_file)
        else:
            node.new_output_file_opt(seg, '.hdf', '--output-file', tags=['SAMPLES'])

        if create_bank:
            node.new_output_file_opt(seg, '.hdf', '--bank-file', tags=['BANK'])

        return node


class CompareExecutable(Executable):
    
    current_retention_level = Executable.MERGED_TRIGGERS
    def create_node(self, samples, bank):
        node = Node(self)
        node.add_input_opt('--input-file', training_samples)
        node.add_input_opt('--bank-file', bank)
        node.new_output_file_opt(samples.segment, '.hdf', '--output-file')
        return node


class PrepareSamplesExecutable(Executable):
    
    current_retention_level = Executable.MERGED_TRIGGERS
    def create_node(self, samples):
        node = Node(self)
        node.add_input_opt('--sample-file', samples)
        node.new_output_file_opt(samples.segment, '.hdf', '--training-file', tags=['TRAINING'])
        node.new_output_file_opt(samples.segment, '.hdf', '--validation-file', tags=['VALIDATION'])
        return node


class InitialiseExecutable(Executable):
    
    current_retention_level = Executable.MERGED_TRIGGERS
    def create_node(self, training_samples, validation_samples, bank, config):
        node = Node(self)
        node.add_input_opt('--training-sample-file', training_samples)
        node.add_input_opt('--validation-sample-file', validation_samples)
        node.add_input_opt('--bank-file', bank)
        node.add_input_opt('--config-file', config)
        node.new_output_file_opt(training_samples.segment, '.hdf', '--output-file')
        return node


class TrainingExecutable(Executable):
    
    current_retention_level = Executable.MERGED_TRIGGERS
    def create_node(self, epochs, training_samples, validation_samples,
                    bank, config, checkpoint=None):
        node = Node(self)
        node.add_opt('--epochs', epochs)
        node.add_input_opt('--training-sample-file', training_samples)
        node.add_input_opt('--validation-sample-file', validation_samples)
        node.add_input_opt('--bank-file', bank)
        node.add_input_opt('--config-file', config)
        if checkpoint is not None:
            node.add_input_opt('--checkpoint-file', checkpoint)
        node.new_output_file_opt(training_samples.segment, '.hdf', '--output-file')
        return node


class StageoutExecutable(Executable):
    
    current_retention_level = Executable.MERGED_TRIGGERS
    def create_node(self, infile, outfile):
        node = Node(self)
        node.add_opt('--input-file', infile.storage_path)
        node.add_output_opt('--output-file', outfile)
        return node


class MergeModelsExecutable(Executable):
    
    current_retention_level = Executable.FINAL_RESULT
    def create_node(self, models):
        node = Node(self)
        seg = models.get_times_covered_by_files()
        node.add_input_list_opt('--model-files', models)
        node.new_output_file_opt(seg, '.hdf', '--output-file')
        return node
