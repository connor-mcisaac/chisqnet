from pycbc.workflow.core import Executable, Node


class FilterTriggersExecutable(Executable):

    current_retention_level = Executable.ALL_TRIGGERS
    def create_node(self, single_det, injfind, inj_single_det,
                    bank, segment, veto):
        node = Node(self)
        node.add_input_list_opt('--trigger-files', single_det)
        node.add_input_list_opt('--inj-find-files', injfind)
        node.add_raw_arg('--inj-trigger-files')
        node.add_raw_arg(' ')
        for group in inj_single_det:
            for infile in group[:-1]:
                node.add_raw_arg(infile.name)
                node.add_raw_arg(',')
                node._add_input(infile)
            node.add_raw_arg(group[-1].name)
            node._add_input(group[-1])
            node.add_raw_arg(' ')
        node.add_input_opt('--bank-file', bank)
        node.add_input_opt('--segment-files', segment)
        node.add_input_opt('--veto-files', veto)
        node.new_output_file_opt(segment.segment, '.hdf', '--output-sample-file',
                                 tags=['TRIGGERS'])
        node.new_output_file_opt(segment.segment, '.hdf', '--output-injection-file',
                                 tags=['INJECTIONS'])
        return node


class PlanningExecutable(Executable):
    
    current_retention_level = Executable.MERGED_TRIGGERS
    def create_node(self, triggers, injections, segments):
        node = Node(self)
        node.add_input_list_opt('--sample-files', triggers)
        node.add_input_list_opt('--injection-files', injections)
        node.add_input_list_opt('--segment-files', segments)
        seg = segments.get_times_covered_by_files()
        node.new_output_file_opt(seg, '.hdf', '--output-sample-file',
                                 tags=['TRIGGERS'])
        node.new_output_file_opt(seg, '.hdf', '--output-injection-file',
                                 tags=['INJECTIONS'])
        node.new_output_file_opt(seg, '.xml', '--output-segment-file',
                                 tags=['SEGMENTS'])
        return node


class PreprocessStrainExecutable(Executable):

    current_retention_level = Executable.ALL_TRIGGERS
    def create_node(self, triggers, injections, segments, job, factor):
        node = Node(self)
        node.add_input_opt('--sample-file', triggers)
        node.add_input_opt('--injection-file', injections)
        node.add_input_opt('--segment-file', segments)
        node.add_opt('--job-num', job)
        node.add_opt('--num-jobs', factor)
        node.new_output_file_opt(triggers.segment, '.hdf', '--output-file')
        return node


class MergeSamplesExecutable(Executable):
    
    current_retention_level = Executable.MERGED_TRIGGERS
    def create_node(self, samples):
        node = Node(self)
        seg = samples.get_times_covered_by_files()
        node.add_input_list_opt('--sample-files', samples)
        node.new_output_file_opt(seg, '.hdf', '--output-file')
        return node


class TrainingExecutable(Executable):
    
    current_retention_level = Executable.MERGED_TRIGGERS
    def create_node(self, samples, bank):
        node = Node(self)
        node.add_input_opt('--sample-file', samples)
        node.add_input_opt('--bank-file', bank)
        node.new_output_file_opt(samples.segment, '.hdf', '--output-file')
        return node
