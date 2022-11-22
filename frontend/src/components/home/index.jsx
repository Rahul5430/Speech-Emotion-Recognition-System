import { useState } from 'react';
import axios from 'axios';
import DragDropFile from './DragDropFile';
axios.defaults.baseURL = process.env.REACT_APP_FLASK_URL;

const Home = () => {
	const [audioFile, setAudioFile] = useState(null);
	const [loading, setLoading] = useState(false);
	const [emotion, setEmotion] = useState('');
	const [clicked, setClicked] = useState(false);
	console.log(process.env.REACT_APP_FLASK_URL);

	const predictFile = (e) => {
		setLoading(true);
		setClicked(true);
		e.preventDefault();
		const data = new FormData();
		data.append('audioFile', audioFile);
		console.log(audioFile);
		axios
			.post('/api', data)
			.then((res) => {
				console.log(res.data);
				setEmotion(res.data);
			})
			.catch((err) => {
				console.log(err);
			})
			.finally(() => {
				setLoading(false);
			});
	};

	return (
		<div className='flex flex-col justify-center items-center text-center h-full'>
			{/* <form
				className='text-center'
				onSubmit={predictFile}
				encType='multipart/form-data'
			>
				<input
					type='file'
					name='audioFile'
					onChange={(e) => setAudioFile(e.target.files[0])}
					required
				/>
				<button type='submit'>Predict</button>
			</form> */}
			<DragDropFile
				predictFile={predictFile}
				setAudioFile={setAudioFile}
			/>
			{!loading && clicked && (
				<h2 className='mt-12'>Predicted emotion is: {emotion}</h2>
			)}
		</div>
	);
};

export default Home;
