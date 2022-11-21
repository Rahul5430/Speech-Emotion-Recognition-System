import { useState, useRef } from 'react';
import './DragDropFile.css';

const DragDropFile = ({ predictFile, setAudioFile }) => {
	const [dragActive, setDragActive] = useState(false);
	const inputRef = useRef(null);

	const handleDrag = (e) => {
		e.preventDefault();
		e.stopPropagation();
		if (e.type === 'dragenter' || e.type === 'dragover') {
			setDragActive(true);
		} else if (e.type === 'dragleave') {
			setDragActive(false);
		}
	};

	const handleDrop = (e) => {
		e.preventDefault();
		e.stopPropagation();
		setDragActive(false);
		if (e.dataTransfer.files && e.dataTransfer.files[0]) {
			// at least one file has been dropped so do something
			// handleFiles(e.dataTransfer.files);
			console.log(e.dataTransfer.files[0]);
		}
	};

	const handleChange = (e) => {
		e.preventDefault();
		if (e.target.files && e.target.files[0]) {
			setAudioFile(e.target.files[0]);
		}
	};

	const onButtonClick = () => {
		inputRef.current.click();
	};

	return (
		<form
			id='form-file-upload'
			onDragEnter={handleDrag}
			// onSubmit={(e) => e.preventDefault()}
			className='text-black'
			encType='multipart/form-data'
            onSubmit={predictFile}
		>
			<input
				ref={inputRef}
				type='file'
				id='input-file-upload'
				multiple={true}
				onChange={handleChange}
                required
			/>
			<label
				id='label-file-upload'
				htmlFor='input-file-upload'
				className={dragActive ? 'drag-active' : ''}
			>
				<div>
					<p>Drag and drop your file here or</p>
					<button className='upload-button' onClick={onButtonClick}>
						Upload a file
					</button>
				</div>
			</label>
			{dragActive && (
				<div
					id='drag-file-element'
					onDragEnter={handleDrag}
					onDragLeave={handleDrag}
					onDragOver={handleDrag}
					onDrop={handleDrop}
				></div>
			)}
			<button type='submit' className='text-white'>
				Predict
			</button>
		</form>
	);
};

export default DragDropFile;
